import fDerivedVars as fdv
import numpy as np
import pandas as pd
import datetime as dt
from pytz import timezone
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


class GliderData:
    def __init__(self, filename, *args, **kwargs):

        self.filename = filename

        print('loading %s...' % filename)
        self.df = pd.read_csv(filename, index_col=False)
        print('done.')

        # glider time
        self.time_par = 'm_present_time'  # POSIX timestamp

        # ctd parameter names
        self.ctd_time_par = 'sci_ctd41cp_timestamp'  # POSIX timestamp
        self.press_par = 'sci_water_pressure'
        self.cond_par = 'sci_water_cond'
        self.temp_par = 'sci_water_temp'
        self.chlor_par = 'sci_flbbcd_chlor_units'

        # gps parameter names
        self.raw_lat_par = 'm_lat'
        self.raw_long_par = 'm_lon'
        self.lat_par = 'latitude'
        self.long_par = 'longitude'
        self.clean_gps_data()

        # derived variable names
        self.depth_par = 'depth'
        self.density_par = 'density'
        self.nsquared_par = 'Nsquared'
        self.epsilon_par = 'epsilon1'
        # calculate derived variables
        self.get_depth()
        self.get_density()
        # self.get_nsquared()

        # flight variables
        self.pitch_par = 'm_pitch'
        self.clean_flight_vars()

        # get yo numbers for individual tracks
        self.inflection_num_par = 'm_tot_num_inflections'
        self.yo_num_par = 'yo_num'
        self.get_yo_nums()

        # labels for plotting of parameters
        self.label_dict = {self.press_par: 'pressure [bar]',
                           self.temp_par: 'T [$^\circ$C]',
                           self.temp_par + '_anomaly': 'T anomaly [$^\circ$C]',
                           self.cond_par: 'conductivity [S/m]',
                           self.chlor_par: 'Chl-a fluorescence [$\mu$g/L]',
                           self.depth_par: 'depth [m]',
                           self.density_par: 'density [kg/m$^3$]',
                           self.nsquared_par: 'N$^2$ [s$^{-2}$]',
                           self.epsilon_par: r'$\varepsilon$ [W/kg]'}

        # datetime and datenum columns
        self.dt_par = 'date'
        self.ctd_dt_par = 'ctd_date'
        self.datenum_par = 'datenum'
        self.ctd_datenum_par = 'ctd_datenum'

        # sets timezone attribute and creates datetime/datenum cols
        self.set_timezone('UTC')
        self.set_date_range(min_t='1990-01-01', max_t='2100-12-31')

        # set datenum as index (m_present_time converted from POSIX --> datenum)
        self.df.index = self.df[self.datenum_par]

        # drop rows without glider time par
        self.df = self.df.dropna(subset=[self.time_par])

        # values to always grab using self.get() method
        self.always_grab = [self.ctd_datenum_par, self.ctd_dt_par, self.ctd_time_par, self.yo_num_par, self.pitch_par, self.depth_par]

        # figure object for plotting
        self.fig, self.axes = None, None

    def bin_avg_p(self, par, binsize):
        """Bin average par based on pressure"""
        print('averaging %s with %.2f bar bins...' % (par, binsize))
        self.df['bin'] = np.floor(self.df[self.press_par] / binsize) * binsize
        par_mean = self.df[['bin', par]].groupby('bin').mean()
        self.df[par + '_mean'] = [par_mean.loc[i][0] if not np.isnan(i) else np.nan for i in self.df['bin']]
        self.df[par + '_anomaly'] = self.df[par] - self.df[par + '_mean']
        return par_mean

    def vert_demean(self, par):
        pass  # ***TODO

    def clean_gps_data(self):
        # convert lat/long from DDMM.MMM to decimal degrees
        self.df[self.lat_par] = self.df[self.raw_lat_par].apply(fdv.convert_latlong)
        self.df[self.long_par] = self.df[self.raw_long_par].apply(fdv.convert_latlong)

        # interpolate values along time axis
        print('interpolating latitude/longitude...')
        self.df[self.lat_par] = self.df[self.lat_par].interpolate(method='index')
        self.df[self.long_par] = self.df[self.long_par].interpolate(method='index')

    def clean_flight_vars(self):
        # linearly interpolate pitch based on time
        print('interpolating measured glider pitch...')
        self.df[self.pitch_par] = self.df[self.pitch_par].interpolate(method='index')

    def get_turbulence_data(self, filename):
        # read in microrider data (*** make function to process turbulence data once I get details)
        print('reading turbulence data...')
        self.turb_df = pd.read_csv(filename)
        self.turb_df.columns = [self.datenum_par, 'p_dbar', 'T', 'Tprime', 'epsilon1', 'epsilon2']
        self.turb_df.index = self.turb_df[self.datenum_par]
        # concatenate with rest of data
        print('concatenating...')
        self.df = pd.concat([self.df, self.turb_df], sort=True)
        self.df = self.df.sort_index()

        # interpolate to rest of ctd_timestamp data
        print('interpolating...')
        for col in self.turb_df.columns:
            self.df[col] = self.df[col].interpolate(method='nearest')

    def get(self, par, **kwargs):
        """
        Get dataframe for CTD time and par, subset over data range.
        :param par: str or list. If list of par names, gets all pars in addition to timestamp and datetime.
        :param min_val: float or list, minimum value of par(s).
        :param max_val: float or list, maximum value of par(s).

        :return: dataframe containing data for par subset to range
        """

        if 'min_val' in kwargs.keys() or 'max_val' in kwargs.keys():
            min_val = kwargs['min_val']
            max_val = kwargs['max_val']
        elif type(par) == list:
            min_val = [-np.inf] * len(par)
            max_val = [np.inf] * len(par)
        else:
            min_val = -np.inf
            max_val = np.inf

        if type(par) == list:
            for p in par:
                if p not in self.df.columns:
                    raise Exception('ERROR: parameter %s not valid.' % p)
            if type(min_val) != list or type(max_val) != list:
                raise Exception('ERROR: min_val and max_val must be lists of values for each parameter argument.')

        else:
            if par not in self.df.columns:
                raise Exception('ERROR: parameter %s not valid.' % par)

        # get self.always_grab pars and input par, subset to data range for time and input par
        if type(par) == list:
            df = self.df[{*self.always_grab, *par}]
            for i, p in enumerate(par):
                df = df[df[p].between(min_val[i], max_val[i]) & df[self.ctd_time_par].between(self.min_ts, self.max_ts)]
        else:
            df = self.df[{*self.always_grab, par}]
            df = df[df[par].between(min_val, max_val) & df[self.ctd_time_par].between(self.min_ts, self.max_ts)]

        return df

    def get_depth(self):
        print('calculating depth...')
        self.df[self.depth_par] = fdv.calculate_depth(self.df[self.press_par],
                                                      self.df[self.lat_par])
        return self.df[self.depth_par]

    def get_density(self):
        print('calculating density...')
        self.df[self.density_par] = fdv.calculate_density(self.df[self.temp_par],
                                                          self.df[self.press_par],
                                                          self.df[self.cond_par],
                                                          self.df[self.lat_par],
                                                          self.df[self.long_par])
        return self.df[self.density_par]

    def get_nsquared(self):
        print('calculating N^2...')
        self.df[self.nsquared_par] = fdv.calculate_n2(self.df[self.temp_par],
                                                      self.df[self.press_par],
                                                      self.df[self.cond_par],
                                                      self.df[self.lat_par],
                                                      self.df[self.long_par])
        return self.df[self.nsquared_par]

    def get_yo_nums(self):
        """Creates parameter for current yo number for counting/plotting individual tracks"""
        print('getting yo numbers...')
        self.df[self.yo_num_par] = fdv.calculate_yo_num(self.df[self.inflection_num_par])
        return self.df[self.yo_num_par]

    def get_dist_traveled(self, yo_df):
        """Gets horizontal distance traveled using pitch and depth for an individual yo"""
        x_dist = fdv.calculate_dist_traveled(yo_df[self.pitch_par], yo_df[self.depth_par])
        return x_dist

    def get_label(self, par):
        """Get parameter label for plotting"""
        if par in self.label_dict.keys():
            par_name = self.label_dict[par]
        else:
            par_name = ' '.join(par.split('_')[1:])
        return par_name

    def get_datetime_plot_label(self, dts, times=False):

        if times:  # used for plotting individual yos (just show start day and start/end times)
            d_range = min(dts).strftime('%b %d %Y, %H:%M-') + max(dts).strftime('%H:%M')
            return d_range

        utc_offset = int(min(dts).utcoffset().total_seconds() / 60 / 60)
        utc_offset = '%i:00' % utc_offset if utc_offset < 0 else '+%i:00' % utc_offset
        if min(dts).year == max(dts).year:
            if min(dts).month == max(dts).month:
                if min(dts).day == max(dts).day:
                    # all data spans single day
                    d_range = min(dts).strftime('%b %d %Y')
                else:
                    # multiple days but same month
                    d_range = min(dts).strftime('%b %d-') + max(dts).strftime('%d %Y')
            else:
                # multiple months but same year
                d_range = min(dts).strftime('%b %d-') + max(dts).strftime('%b %d %Y')
        else:
            # multiple years
            d_range = min(dts).strftime('%b %d %Y-') + max(dts).strftime('%b %d %Y')

        label = 'Local Time (UTC%s)\n%s' % (utc_offset, d_range)
        return label

    def get_info(self):
        for col in self.df.columns:
            parmin, parmax = min(self.df[col]), max(self.df[col])
            print('%s: min: %.2f, max: %.2f' % (col, parmin, parmax))

    def interpolate_contour_grid(self, x, y, z, res=(4000, 1000), method='nearest'):
        # grid resolution for interpolation
        x_range = [min(x), max(x)]
        y_range = [min(y), max(y)]

        xi = np.linspace(*x_range, res[0])
        yi = np.linspace(*y_range, res[1])

        grid_x, grid_y = np.meshgrid(xi, yi)
        zi = griddata((np.array(x), np.array(y)), np.array(z), (grid_x, grid_y), method=method)

        return xi, yi, zi

    def set_date_range(self, min_t, max_t):
        """Set date range to subset all data from when using self.get() method"""
        self.min_t = min_t
        self.max_t = max_t
        self.min_ts = self.str_to_ts(min_t)
        self.max_ts = self.str_to_ts(max_t)

    def set_timezone(self, tz):
        self.tz = timezone(tz)

        # convert timestamps to datetime objects
        self.df[self.dt_par] = [self.ts_to_dt(ti) for ti in self.df[self.time_par]]
        self.df[self.ctd_dt_par] = [self.ts_to_dt(ti) for ti in self.df[self.ctd_time_par]]

        # convert timestamps to datenum format
        self.df[self.datenum_par] = [self.ts_to_dn(ti) for ti in self.df[self.time_par]]
        self.df[self.ctd_datenum_par] = [self.ts_to_dn(ti) for ti in self.df[self.ctd_time_par]]

    def str_to_ts(self, s):
        # convert date string to timestamp
        try:
            return time.mktime(dt.datetime.strptime(s, "%Y-%m-%d %H:%M").astimezone(self.tz).timetuple())
        except:
            return time.mktime(dt.datetime.strptime(s, "%Y-%m-%d").astimezone(self.tz).timetuple())

    def ts_to_dt(self, ti):
        # convert timestamp to datetime object
        try:
            val = dt.datetime.fromtimestamp(ti).astimezone(self.tz)
            return val
        except:
            if (not np.isnan(ti)) and ti != 0:
                print('WARNING: Encountered invalid timestamp value: {0}'.format(ti))
            return np.nan

    def ts_to_dn(self, ti):
        # convert timestamp to matlab datenum
        try:
            val = mdates.date2num(dt.datetime.fromtimestamp(ti).astimezone(self.tz))
            return val
        except:
            return np.nan

    def plot(self, par, sep_yos=False, plot_index=(1, 1, 1), **kwargs):
        """Plot par as a function of time, subset to data range."""
        print('plotting %s time series...' % par)

        par_list = par if type(par) == list else [par]

        # set data range
        min_val = kwargs['min_val'] if 'min_val' in kwargs.keys() else [-np.inf] * len(par_list)
        max_val = kwargs['max_val'] if 'max_val' in kwargs.keys() else [np.inf] * len(par_list)

        df = self.get(par_list, min_val=min_val, max_val=max_val)

        if sep_yos:
            # separate plot for each yo
            for yo_num in set(df[self.yo_num_par]):
                print('plotting yo num. %s...' % yo_num)
                yo_df = df[df[self.yo_num_par] == yo_num]
                # QC measure: filter out sections staying at surface, where pitch is small
                print('filtering out track sections with pitch deviation >0.1 from median...')
                med_pitch = yo_df[self.pitch_par].median()
                yo_df = yo_df[abs(yo_df[self.pitch_par] - med_pitch) < 0.1]
                x_dist = self.get_dist_traveled(yo_df)
                # create Nx1 subplots
                fig, axes = plt.subplots(len(par_list), 1, sharex=True, squeeze=False)
                for i, p in enumerate(par_list):
                    ax = axes[i][0]
                    if not ((min_val[i] == -np.inf) or (max_val[i] == np.inf)):
                        ax.set(ylim=(min_val[i], max_val[i]))
                    if p in [self.press_par, self.depth_par]:
                        ax.invert_yaxis()
                    ax.plot(x_dist, yo_df[p])
                    ax.set(ylabel=self.get_label(p))
                ax.set(xlabel='Horizontal distance [m]')
                fig.suptitle(self.get_datetime_plot_label(yo_df[self.ctd_dt_par], times=True))
                fig.savefig(f'.\\figs\\yos\\{int(yo_num)}.png')
                plt.close(fig)

        else:
            # create Nx1 subplots
            fig, axes = plt.subplots(len(par_list), 1, sharex=True, squeeze=False)
            for i, p in enumerate(par_list):
                ax = axes[i][0]
                ax.plot(df[self.ctd_dt_par], df[p])
                ax.set(ylabel=self.get_label(p))
            plt.show()
        print('done.')

    def plot_xy(self, par, contour=False, log_scale=False, plot_index=(1, 1, 1), **kwargs):
        """Plot par by color as function of time and pressure/depth"""
        print('plotting %s profile...' % par)
        # set data range
        min_val = kwargs['min_val'] if 'min_val' in kwargs.keys() else [-np.inf] * 2
        max_val = kwargs['max_val'] if 'max_val' in kwargs.keys() else [np.inf] * 2

        # get time, pressure, and par (min/max_vals for pressure and par, respectively)
        df = self.get([self.press_par, par], min_val=min_val, max_val=max_val)
        df['press_dbar'] = df[self.press_par] * 10

        # set point size
        s = kwargs['s'] if 's' in kwargs.keys() else 1
        # determine colormap and colormap range
        cmap = kwargs['cmap'] if 'cmap' in kwargs.keys() else 'inferno'
        vmin, vmax = kwargs['c_range'] if 'c_range' in kwargs.keys() else [min(df[par]), max(df[par])]
        # colormap scaling
        norm = colors.LogNorm(vmin=vmin, vmax=vmax) if log_scale else colors.Normalize(vmin=vmin, vmax=vmax)

        # plot
        # if first plot, get corresponding axis
        if plot_index[2] == 1:
            self.fig, self.axes = plt.subplots(*plot_index[:2], sharex=True, sharey=True, squeeze=False)
        col_num = plot_index[2] % plot_index[1]
        row_num = int((plot_index[2] - col_num)/plot_index[1] - 1)
        self.ax = self.axes[row_num, col_num]
        datenum = df[self.ctd_datenum_par]
        # x axis formatting (if last subplot)
        if plot_index[0] * plot_index[1] == plot_index[2]:
            self.ax.set(xlabel=self.get_datetime_plot_label(df[self.ctd_dt_par]))
            self.ax.set_xlim(min(datenum), max(datenum))
            self.fig.autofmt_xdate()  # rotates xtick labels
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=self.tz))
            self.ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18], tz=self.tz))
        # y axis formatting
        self.ax.set(ylabel='pressure [dbar]')
        self.ax.set_ylim(0, max(df['press_dbar']))
        self.ax.invert_yaxis()
        # add data to plot
        if contour:
            res = kwargs['res'] if 'res' in kwargs.keys() else (4000, 1000)
            method = kwargs['method'] if 'method' in kwargs.keys() else 'nearest'
            levels = kwargs['levels'] if 'levels' in kwargs.keys() else 20
            sc = self.ax.contourf(*self.interpolate_contour_grid(datenum, df['press_dbar'], df[par], res=res, method=method),
                             cmap=cmap, norm=norm, levels=levels)
        else:
            sc = self.ax.scatter(datenum, df['press_dbar'], s=s, c=df[par], cmap=cmap, norm=norm)
        # add color bar
        cb = self.fig.colorbar(sc, ax=self.ax)
        cb.set_label(self.get_label(par))
        # show if last subplot, and reset figure
        if plot_index[0] * plot_index[1] == plot_index[2]:
            plt.show()
            self.fig, self.axes = None, None
        print('done.')

    def plot_3d_track(self, par, track='all', contour=False, **kwargs):
        """Makes 3D plot of tracks using lat, long time

        :param par: parameter to plot
        :param track: subset of all data tracks to plot.
        :param contour: if True, plots contours in addition to data

        :return: 3d plot
        """
        print('plotting %s 3D profile...' % par)
        print('Using tracks: %s' % track)
        # set data range for lat, long, pressure and par
        min_val = kwargs['min_val'] if 'min_val' in kwargs.keys() else [-np.inf] * 4
        max_val = kwargs['max_val'] if 'max_val' in kwargs.keys() else [np.inf] * 4

        df = self.get([self.lat_par, self.long_par, self.press_par, par], min_val=min_val, max_val=max_val)
        df['press_dbar'] = df[self.press_par] * 10

        # make 3D plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.invert_zaxis()
        ax.scatter(df[self.lat_par], df[self.long_par], df['press_dbar'], c=df[par])
        plt.show()

