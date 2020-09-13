import numpy as np
import pandas as pd
import datetime as dt
from pytz import timezone
import time
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class GliderData:
    def __init__(self, filename, *args, **kwargs):

        self.filename = filename

        print('loading %s...' % filename)
        self.df = pd.read_csv(filename, index_col=False)
        print('done.')

        # ctd parameter names
        self.ctd_time_par = 'sci_ctd41cp_timestamp'
        self.press_par = 'sci_water_pressure'
        self.cond_par = 'sci_water_cond'
        self.temp_par = 'sci_water_temp'

        self.label_dict = {self.press_par: 'pressure [bar]',
                           self.temp_par: 'T [$^\circ$C]'}

        self.tz = timezone('UTC')
        self.set_date_range(min_t='1990-01-01', max_t='2100-12-31')

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


    def get_label(self, par):
        """Get parameter label for plotting"""
        if par in self.label_dict.keys():
            par_name = self.label_dict[par]
        else:
            par_name = ' '.join(par.split('_')[1:])
        return par_name

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
        df = self.get(par, min_val=min_val, max_val=max_val)

        # plot
        fig, ax = plt.subplots()
        ax.plot(df['date'], df[par])
        ax.set(ylabel=self.get_label(par))
        plt.show()

    def plot_xy(self, par, contour=False, **kwargs):
        """Plot par by color as function of time and pressure/depth"""
        # set data range
        min_val = kwargs['min_val'] if 'min_val' in kwargs.keys() else [-np.inf] * 2
        max_val = kwargs['max_val'] if 'max_val' in kwargs.keys() else [np.inf] * 2

        # get time, pressure, and par
        df = self.get([self.press_par, par], min_val=min_val, max_val=max_val)

        # set point size
        s = kwargs['s'] if 's' in kwargs.keys() else 1

        # determine colormap range
        if 'c_range' in kwargs.keys():
            vmin, vmax = kwargs['c_range']
        else:
            vmin, vmax = min(df[par]), max(df[par])

        # plot
        fig, ax = plt.subplots()
        # x axis formatting
        ax.set(xlabel='Local Time')
        ax.set_xlim(min(df['date']), max(df['date']))
        locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_formatter(formatter)
        # y axis formatting
        ax.set(ylabel='pressure [bar]')
        ax.set_ylim(0, max(df[self.press_par]))
        ax.invert_yaxis()
        # add data to plot
        if contour:
            norm = colors.Normalize(vmin, vmax)
            sc = ax.tricontourf(df['date'], df[self.press_par], df[par], norm=norm)
        else:
            sc = ax.scatter(df['date'], df[self.press_par], s=s, c=df[par], cmap='inferno', vmin=vmin, vmax=vmax)
        # add color bar
        cb = fig.colorbar(sc)
        cb.set_label(self.get_label(par))
        plt.show()
