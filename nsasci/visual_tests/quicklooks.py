import pathlib
import sqlite3
import plt_tools
import xarray as xr
from atmPy.data_archives import arm
import datetime
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class Data(object):
    def __init__(self, kazr=True, ceilometer=True, timezone = -9, path2database = None):
        self.timezone = timezone
        self.send_message = lambda x: print(x)
        self.path2database = pathlib.Path(path2database)
        self.db_table_name = "vis_nsascience.0.1"
        self.path2nsascience = pathlib.Path(
            '/mnt/telg/data/arm_data/OLI/tethered_balloon/nsascience/nsascience_avg1min/')

        self.path2kazr = pathlib.Path('/mnt/telg/data/arm_data/OLI/kazr_1min/')
        self.path2ceil = pathlib.Path('/mnt/telg/data/arm_data/OLI/ceilometer_1min/')

        # path2quicklooks = pathlib.Path('/mnt/telg/data/arm_data/OLI/tethered_balloon/nsascience/nsascience_quicklook/')

        # get some database stuff
        qu = "SELECT * FROM instrumet_performence_POPS_by_file"
        with sqlite3.connect(self.path2database) as db:
            self.inst_perform_pops = pd.read_sql(qu, db)

        self.cm_meins = plt_tools.colormap.my_colormaps.relative_humidity(reverse=True)
        self.cm_meins.set_bad('red')
        self.lw_pgc = 3

        # Preparations
        ##nasascience
        df_nsa = pd.DataFrame(list(self.path2nsascience.glob('*.nc')), columns=['path'])
        df_nsa['start_datetime'] = df_nsa.path.apply(lambda x: self.path2startdatetime(x, 1))
        # df_nsa['start_datetime_AKst'] = df_nsa.start_datetime + pd.to_timedelta(self.timezone, unit='h')
        df_nsa.sort_values('start_datetime', inplace=True)
        # df_nsa['start_date_AKst'] = pd.DatetimeIndex(df_nsa.start_datetime_AKst).date
        df_nsa['end_datetime'] = df_nsa.path.apply(lambda x: self.load_nsascience(x, return_end=True))

        # self.stakunique = np.unique(df_nsa.start_datetime)
        # self.valid_dates = self.stakunique
        self.valid_dates = np.sort(np.unique(np.concatenate((pd.DatetimeIndex(df_nsa.start_datetime + np.timedelta64(self.timezone, 'h')).date, pd.DatetimeIndex(df_nsa.end_datetime+ np.timedelta64(self.timezone, 'h')).date))))# + np.timedelta64(self.timezone, 'h')
        self.valid_dates = pd.to_datetime(self.valid_dates)
        self.df_nsa = df_nsa

        ## kazr

        paths_kazr = list(self.path2kazr.glob('*'))
        paths_kazr.sort()

        df_kzar = pd.DataFrame(paths_kazr, columns=['paths'])
        df_kzar['names'] = [path.name for path in paths_kazr]
        if not kazr:
            df_kzar = df_kzar.iloc[:1]
        df_kzar['start_datetime'] = df_kzar.paths.apply(lambda x: self.path2startdatetime(x, 2))
        df_kzar['end_datetime'] = df_kzar.paths.apply(lambda x: self.load_kazr(x, return_end=True))

        self.df_kzar = df_kzar

        ## Ceil
        paths_ceil = list(self.path2ceil.glob('*.nc'))
        paths_ceil.sort()

        df_ceil = pd.DataFrame(paths_ceil, columns=['paths'])
        df_ceil['names'] = [path.name for path in paths_ceil]
        if not ceilometer:
            df_ceil = df_ceil.iloc[:1]
        df_ceil['start_datetime'] = df_ceil.paths.apply(lambda x: self.path2startdatetime(x, 2))
        df_ceil['end_datetime'] = df_ceil.paths.apply(
            lambda x: arm.read_ceilometer_nc(x.as_posix()).cloudbase.get_timespan()[-1])
        self.df_ceil = df_ceil

    # helpers

    def load_kazr(self, path, return_end=False
                  #               avg = (1,'m')
                  ):
        kz = arm.read_kazr_nc(path.as_posix(), timezone = self.timezone)
        #     kz = kz.average_time(avg)
        if return_end:
            return kz.reflectivity.get_timespan()[-1]
        else:
            return kz

    def load_nsascience(self, path, return_end=False):
        ds = xr.open_dataset(path)
        #         dsrs = ds.resample(datetime='1min').mean()
        #         dsrs.attrs = ds.attrs
        if return_end:
            return ds.datetime.values[-1]
        else:
            return ds

    def path2startdatetime(self, path, idx):
        date, time = path.name.split('.')[idx:idx + 2]
        start_datetime = pd.to_datetime('{} {}'.format(date, time))
        return start_datetime

    def plot_pops_NC(self, df_nsa_aotd, a, resample='1min'):
        outs = []
        returns = {}
        i = 0
        for e, meas in enumerate(df_nsa_aotd.data):
            # test and skip bad quality values
            if meas.based_on_file_pops in self.inst_perform_pops.fname.values:
                qualtest = self.inst_perform_pops[self.inst_perform_pops.fname == meas.based_on_file_pops]
                if qualtest.quality.loc[0] == 'bad':
                    self.send_message('bad quality measurement skiped ({})'.format(meas.based_on_file_pops))
                    continue

            #             if resample:
            #                 meas = meas.resample(datetime=resample).mean()

            colorbar = False
            a, lc, cm = plt_tools.plot.plot_gradiant_color(meas.datetime.values + np.timedelta64(self.timezone, 'h'),
                                                           meas.altitude.values,
                                                           meas.pops_particle_number_concentration.values,
                                                           ax=a,
                                                           colorbar=colorbar)

            out = dict(lc=lc, cm=cm)
            out['mean'] = float(meas.pops_particle_number_concentration.median().values)
            out['std'] = float(meas.pops_particle_number_concentration.std().values)
            #             out['cmax'] = out['mean'] + (1 *  out['std'])

            #             meast = meas.resample(datetime = '10min').mean()
            # trying to get rid of those plumes close to the ground
            meast = meas.copy(deep=True)
            meast.pops_particle_number_concentration[meast.altitude < 20] = np.nan
            meast.pops_particle_number_concentration[np.isnan(meast.altitude)] = np.nan
            out['cmax'] = meast.pops_particle_number_concentration.max()
            out['cmin'] = meast.pops_particle_number_concentration.min()
            out['alt_max'] = meas.altitude.max()
            outs.append(out)
            i += 1

        cmax = max([out['cmax'] for out in outs])
        cmin = min([out['cmin'] for out in outs])
        returns['clim'] = (cmin, cmax)
        lcs = [out['lc'] for out in outs]
        # returns['zobjects'] = lcs
        a.zobjects = lcs
        for lc in lcs:
            lc.set_clim(cmin, cmax)
            #             lc.set_clim(0,25)
            lc.set_cmap(self.cm_meins)
            lc.set_linewidth(self.lw_pgc)

        a.set_ylim(-10, max([out['alt_max'] for out in outs]) * 1.2)

        a.xaxis.set_major_formatter(plt.DateFormatter("%H:%M:%S"))
        # a.set_xlabel('')

        f = a.get_figure()
        f.autofmt_xdate()
        a.set_ylabel('Altitude (m)')

        # colorbar

        cb, cax = plt_tools.colorbar.colorbar_axis_split_off(lc, a)
        # self.lc = lc
        # self.cb, self.cax = cb, cax
        a.cax = cax
        cb.locator = plt.MaxNLocator(5, prune='both')
        cb.update_ticks()
        cax.set_ylabel('NC$_{POPS}$ (#/cm$^3$)', labelpad=0.5)
        returns['a'] = a
        return returns

    def plot_temp(self, df_nsa_aotd, a, resample='1min'):
        outs = []
        returns = {}
        i = 0
        for e, meas in enumerate(df_nsa_aotd.data):
            # test and skip bad quality values
            if meas.based_on_file_pops in self.inst_perform_pops.fname.values:
                qualtest = self.inst_perform_pops[self.inst_perform_pops.fname == meas.based_on_file_pops]
                if qualtest.quality.loc[0] == 'bad':
                    print('bad quality measurement skiped ({})'.format(meas.based_on_file_pops))
                    continue

            #         if resample:
            #             meas = meas.resample(datetime=resample).mean()

            colorbar = False
            a, lc, cm = plt_tools.plot.plot_gradiant_color(meas.datetime.values + np.timedelta64(self.timezone, 'h'),
                                                           meas.altitude.values, meas.temp.values,
                                                           ax=a,
                                                           colorbar=colorbar)

            out = dict(lc=lc, cm=cm)
            out['mean'] = float(meas.temp.median().values)
            out['std'] = float(meas.temp.std().values)
            #         out['cmax'] = out['mean'] + (1 *  out['std'])
            #         out['cmin'] = out['mean'] - (2 *  out['std'])
            out['cmax'] = meas.temp.max()
            out['cmin'] = meas.temp.min()
            out['alt_max'] = meas.altitude.max()
            outs.append(out)
            i += 1

        cmax = max([out['cmax'] for out in outs])
        cmin = min([out['cmin'] for out in outs])
        returns['clim'] = (cmin, cmax)
        lcs = [out['lc'] for out in outs]
        # returns['zobjects'] = lcs
        a.zobjects = lcs
        for lc in lcs:
            lc.set_clim(cmin, cmax)
            lc.set_cmap(self.cm_meins)
            lc.set_linewidth(self.lw_pgc)

        a.set_ylim(-10, max([out['alt_max'] for out in outs]) * 1.2)

        a.xaxis.set_major_formatter(plt.DateFormatter("%H:%M:%S"))
        # a.set_xlabel('')

        f = a.get_figure()
        f.autofmt_xdate()
        a.set_ylabel('Altitude (m)')

        # colorbar

        cb, cax = plt_tools.colorbar.colorbar_axis_split_off(lc, a)
        a.cax = cax
        cb.locator = plt.MaxNLocator(5, prune='both')
        cb.update_ticks()

        cax.set_ylabel('Temp (°C)', labelpad=0.5)
        returns['a'] = a
        return returns

    def plot_rh(self, df_nsa_aotd, a):
        outs = []
        returns = {}
        i = 0
        for e, meas in enumerate(df_nsa_aotd.data):
            # test and skip bad quality values
            if meas.based_on_file_pops in self.inst_perform_pops.fname.values:
                qualtest = self.inst_perform_pops[self.inst_perform_pops.fname == meas.based_on_file_pops]
                if qualtest.quality.loc[0] == 'bad':
                    print('bad quality measurement skiped ({})'.format(meas.based_on_file_pops))
                    continue
            # resample
            #         if resample:
            #             meas = meas.resample(datetime=resample).mean()

            colorbar = False

            values = meas.rh.to_pandas()
            values[values < 0] = np.nan
            values[values > 110] = np.nan
            a, lc, cm = plt_tools.plot.plot_gradiant_color(meas.datetime.values + np.timedelta64(self.timezone, 'h'),
                                                           meas.altitude.values, values,
                                                           ax=a,
                                                           colorbar=colorbar)

            out = dict(lc=lc, cm=cm)
            out['mean'] = float(meas.rh.median().values)
            out['std'] = float(meas.rh.std().values)
            out['cmax'] = meas.rh.max()
            out['cmin'] = meas.rh.min()  # out['mean'] - (2 *  out['std'])
            #         print(out['cmin'])
            out['alt_max'] = meas.altitude.max()
            outs.append(out)
            i += 1

        cmax = max([out['cmax'] for out in outs])
        cmin = min([out['cmin'] for out in outs])
        returns['clim'] = (cmin, cmax)
        lcs = [out['lc'] for out in outs]
        # returns['zobjects'] = lcs
        a.zobjects = lcs
        for lc in lcs:
            lc.set_clim(cmin, cmax)
            lc.set_cmap(self.cm_meins)
            lc.set_linewidth(self.lw_pgc)

        a.set_ylim(-10, max([out['alt_max'] for out in outs]) * 1.2)

        a.xaxis.set_major_formatter(plt.DateFormatter("%H:%M:%S"))
        # a.set_xlabel('')

        f = a.get_figure()
        f.autofmt_xdate()
        a.set_ylabel('Altitude (m)')

        # colorbar

        cb, cax = plt_tools.colorbar.colorbar_axis_split_off(lc, a)
        cax.set_ylabel('RH (%)', labelpad=0.5)

        cb.locator = plt.MaxNLocator(5, prune='both')
        cb.update_ticks()

        a.cax = cax
        returns['a'] = a
        # returns['a'].cax.set_label('buba')

        return returns

    #     set_t = (a,lc,cb)

    def plot(self, date, ax=None
             #              df_nsa_aotd, matches_kazr, matches_ceil, if_output_exists = 'error'
             ):
        """
        if_output_exists: str ['skip', 'overwrite', 'error']"""
        plot_start = date - np.timedelta64(self.timezone, 'h')
        plot_end = plot_start + datetime.timedelta(days=1)
        # find the data first ... maybe outsource this part
        # df_nsa_aotd = self.df_nsa[self.df_nsa.start_date_AKst == date]

        where = np.logical_and(self.df_nsa.end_datetime > plot_start, self.df_nsa.start_datetime < plot_end)
        df_nsa_aotd = self.df_nsa[where]

        df_nsa_aotd['data'] = df_nsa_aotd.path.apply(self.load_nsascience)

        txt = ('opend nsascience files:')
        for path in df_nsa_aotd.path:
            txt +='\n\t{}'.format(path.name)
        self.send_message(txt)

        self.df_nsa_aotd = df_nsa_aotd
        self.date = date

        # find corresponding kazr data
        start_datetime = df_nsa_aotd.start_datetime.min()
        self.df_kzar['td'] = self.df_kzar.start_datetime - start_datetime

        where = np.logical_and(self.df_kzar.end_datetime > plot_start, self.df_kzar.start_datetime < plot_end)
        matches_kazr = self.df_kzar[where]

        matches_kazr['data'] = matches_kazr.paths.apply(self.load_kazr)

        txt = ('opend kzar files:')
        for path in matches_kazr.paths:
            txt +='\n\t{}'.format(path.name)
        self.send_message(txt)

        self.matches_kazr = matches_kazr


        # find corresponding ceilometer data
        self.df_ceil['td'] = self.df_ceil.start_datetime - start_datetime

        where = np.logical_and(self.df_ceil.end_datetime > plot_start, self.df_ceil.start_datetime < plot_end)
        matches_ceil = self.df_ceil[where]
        matches_ceil['data'] = matches_ceil.paths.apply(lambda x: arm.read_ceilometer_nc(x.as_posix(), timezone = self.timezone))
        txt = ('opend ceilometer data files:')
        for path in matches_ceil.paths:
            txt +='\n\t{}'.format(path.name)
        self.send_message(txt)
        ##
        ##
        st = df_nsa_aotd.start_datetime.iloc[0]
        # txt = "{}-{:02d}-{:02d}".format(st.year, st.month, st.day)
        txt = "{}-{:02d}-{:02d}".format(date.year, date.month, date.day)
        #         path_out = path2quicklooks.joinpath('quicklook_{}.png'.format(txt.replace('-','')))
        #         if path_out.exists():
        #             if if_output_exists == 'skip':
        #                 return False
        #             elif if_output_exists == 'overwrite':
        #                 pass
        #             elif if_output_exists == 'error':
        #                 raise FileExistsError('file exists: {}'.format(path_out))
        #             else:
        #                 raise ValueError('{} is not an option for "if_output_exists"'.format(if_output_exists))
        if isinstance(ax, type(None)):
            f, a = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0})
            f.set_figheight(f.get_figheight() * 1.5)
        else:
            a = ax
            f = a[0].get_figure()
            for at in a:
                at.clear()
                try:
                    at.cax.remove()
                except:
                    pass
        atemp, ahr, apops = a

        # for kz in matches_kazr.data:
        #     kz.reflectivity.data.index += np.timedelta64(2, 'h')

        for at in a:
            for kz in matches_kazr.data:
                # kz.reflectivity.data.index += np.timedelta64(self.timezone, 'h')
                kz.reflectivity.plot(snr_max=10, ax=at, zorder=0)

            for ceil in matches_ceil.data:
                ceil.cloudbase.plot(color=colors[1], ax=at, zorder=1)

        plot_pops_content = self.plot_pops_NC(df_nsa_aotd, apops)
        plot_temp_content = self.plot_temp(df_nsa_aotd, atemp)
        plot_rh_content = self.plot_rh(df_nsa_aotd, ahr)

        plt_tools.axes.labels.set_shared_label(a, 'Altitude (m)', axis='y')

        at = a[0]
        at.text(0.05, 0.8, txt, transform=at.transAxes)

        a[-1].set_xlabel('')

        f.patch.set_alpha(0)
        #         f.savefig(path_out, bbox_inches = 'tight')
        out =  {}
        out['temperatur'] = plot_temp_content
        out['relative_humidity'] = plot_rh_content
        out['nc_pops'] = plot_pops_content
        # for at in a:
        #     outd = {}
        #     outd['a'] = at
        #     out.append(outd)
        # for at in a:
        #     a.cax.set_ylabel('buba')
        return out
