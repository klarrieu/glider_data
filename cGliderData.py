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

        # set m_present_time as index
        self.time_par = 'm_present_time'
        # self.df.index = self.df[self.time_par]

        # ctd parameter names
        self.ctd_time_par = 'sci_ctd41cp_timestamp'
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
        self.Nsquared_par = 'Nsquared'
        # calculate derived variables
        self.get_depth()
        self.get_density()
        self.get_Nsquared()

        # labels for plotting of parameters
        self.label_dict = {self.press_par: 'pressure [bar]',
                           self.temp_par: 'T [$^\circ$C]',
                           self.cond_par: 'conductivity [S/m]',
                           self.chlor_par: 'Chlorophyll-a fluorescence [$\mu$g/L]',
                           self.density_par: 'density [kg/m$^3$]',
                           self.Nsquared_par: 'N$^2$ [s$^{-2}$]'}

        self.tz = timezone('UTC')
        self.set_date_range(min_t='1990-01-01', max_t='2100-12-31')

    def clean_gps_data(self):
        # convert lat/long from DDMM.MMM to decimal degrees
        self.df[self.lat_par] = self.df[self.raw_lat_par].apply(fdv.convert_latlong)
        self.df[self.long_par] = self.df[self.raw_long_par].apply(fdv.convert_latlong)

        # interpolate values along time axis
        self.df[self.lat_par] = self.df[self.lat_par].interpolate(method='index')
        self.df[self.long_par] = self.df[self.long_par].interpolate(method='index')

        # only keep values where pressure is also defined
        self.df.loc[pd.isnull(self.df[self.press_par]), [self.lat_par, self.long_par]] = np.nan

        # remove values before first/after last gps points
        self.df.loc[pd.isnull(self.df[self.lat_par]), :] = np.nan

    def get(self, par, **kwargs):
        """
        Get dataframe for time and par, subset over data range.
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

        # get time and par and subset to data range
        if type(par) == list:
            df = self.df[[self.ctd_time_par, *par]]
            for i, p in enumerate(par):
                df = df[df[p].between(min_val[i], max_val[i]) & df[self.ctd_time_par].between(self.min_ts, self.max_ts)]
        else:
            df = self.df[[self.ctd_time_par, par]]
            df = df[df[par].between(min_val, max_val) & df[self.ctd_time_par].between(self.min_ts, self.max_ts)]

        # convert timestamp to datetime object
        df['date'] = [dt.datetime.fromtimestamp(ti).astimezone(self.tz) for ti in df[self.ctd_time_par]]

        return df

    def get_depth(self):
        self.df[self.depth_par] = fdv.calculate_depth(self.df[self.press_par],
                                                      self.df[self.lat_par])
        return self.df[self.depth_par]

    def get_density(self):
        self.df[self.density_par] = fdv.calculate_density(self.df[self.temp_par],
                                                          self.df[self.press_par],
                                                          self.df[self.cond_par],
                                                          self.df[self.lat_par],
                                                          self.df[self.long_par])
        return self.df[self.density_par]

    def get_Nsquared(self):
        self.df[self.Nsquared_par] = fdv.calculate_n2(self.df[self.temp_par],
                                                      self.df[self.press_par],
                                                      self.df[self.cond_par],
                                                      self.df[self.lat_par],
                                                      self.df[self.long_par])
        return self.df[self.Nsquared_par]

    def get_label(self, par):
        """Get parameter label for plotting"""
        if par in self.label_dict.keys():
            par_name = self.label_dict[par]
        else:
            par_name = ' '.join(par.split('_')[1:])
        return par_name

    def get_datetime_plot_label(self, dts):
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

    def str_to_ts(self, s):
        # convert date string to timestamp
        try:
            return time.mktime(dt.datetime.strptime(s, "%Y-%m-%d %H:%M").astimezone(self.tz).timetuple())
        except:
            return time.mktime(dt.datetime.strptime(s, "%Y-%m-%d").astimezone(self.tz).timetuple())

    def plot(self, par, min_val=-np.inf, max_val=np.inf):
        """Plot par as a function of time, subset to data range."""
        print('plotting %s time series...' % par)
        df = self.get(par, min_val=min_val, max_val=max_val)

        # plot
        fig, ax = plt.subplots()
        ax.plot(df['date'], df[par])
        ax.set(ylabel=self.get_label(par))
        plt.show()
        print('done.')

    def plot_xy(self, par, contour=False, **kwargs):
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

        # plot
        fig, ax = plt.subplots()
        datenum = mdates.date2num(df['date'])
        # x axis formatting
        ax.set(xlabel=self.get_datetime_plot_label(df['date']))
        ax.set_xlim(min(datenum), max(datenum))
        fig.autofmt_xdate()  # rotates xtick labels
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=self.tz))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18], tz=self.tz))
        # y axis formatting
        ax.set(ylabel='pressure [dbar]')
        ax.set_ylim(0, max(df['press_dbar']))
        ax.invert_yaxis()
        # add data to plot
        if contour:
            res = kwargs['res'] if 'res' in kwargs.keys() else (4000, 1000)
            method = kwargs['method'] if 'method' in kwargs.keys() else 'nearest'
            levels = kwargs['levels'] if 'levels' in kwargs.keys() else 20
            sc = ax.contourf(*self.interpolate_contour_grid(datenum, df['press_dbar'], df[par], res=res, method=method),
                             cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
        else:
            sc = ax.scatter(datenum, df['press_dbar'], s=s, c=df[par], cmap=cmap, vmin=vmin, vmax=vmax)
        # add color bar
        cb = fig.colorbar(sc)
        cb.set_label(self.get_label(par))
        plt.show()
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

