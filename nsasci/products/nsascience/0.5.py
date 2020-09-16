import pathlib
import sqlite3
import pandas as pd
import numpy as np
import xarray as xr
from metpy import calc
from metpy.units import units
from atmPy.aerosols.size_distribution import sizedistribution as sd
from atmPy.aerosols import size_distribution
from atmPy.data_archives import arm
import scipy as sp
import sys
import plt_tools
import matplotlib.pyplot as plt
import ruptures as rpt
import warnings


def open_iMet(path, verbose=False):
    ds = xr.open_dataset(path)

    # txt = ',\n'.join('"{}"'.format(k) for k in ds.variables.keys())
    # print(txt)

    imet_columns2use = {
    # "datetime",
    "altitude (from iMet PTU) [km]": 'altitude_ptu',
    "iMet pressure [mb]": 'atm_pressure',
    "iMet air temperature (corrected) [deg C]": 'temp',
    # "iMet air temperature (raw) [deg C]",
    "iMet humidity [RH %]": 'rh',
#     "iMet frostpoint [deg C]": 'frost_point',
    # "iMet internal temperature [deg C]",
    # "iMet battery voltage [V]",
    "iMet theta [K]": 'potential_temperature',
    # "iMet temperature (of pressure sensor) [deg C]",
    # "iMet temperature (of humidity sensor) [deg C]",
    # "iMet ascent rate [m*s^-1]",
    # "iMet water vapor mixing ratio [ppmv]",
    # "iMet total column water [mm]",
    # "GPS latitude",
    # "GPS longitude",
    "GPS altitude [km]": 'altitude_gps',
    # "GPS num satellites",
    # "GPS pressure [mb]",
    # "GPS wind speed [m*s^-1]",
    # "GPS wind direction [deg]",
    # "GPS ascent rate [m*s^-1]",
    # "GPS(X) east velocity [m*s^-1]",
    # "GPS(X) north velocity [m*s^-1]",
    # "GPS(X) up velocity [m*s^-1]",
#     "GPS time [h:m:s GMT]": 'time_gps',
    # "GPS heading from launch [deg]",
    # "GPS elevation angle from launch [deg]",
    # "GPS distance from launch [km]",
    # "predicted landing latitude",
    # "predicted landing longitude",
    # "predicted time to landing [min]",
    # "POPS Particle Rate [count]",
    # "POPS Flow Rate [cc*s^-1]",
    # "POPS Temperature [deg C]",
    # "POPS Bin 0",
    # "POPS Bin 1",
    # "POPS Bin 2",
    # "POPS Bin 3",
    # "POPS Bin 4",
    # "POPS Bin 5",
    # "POPS Bin 6",
    # "POPS Bin 7",
    # "POPS Bin 8",
    # "POPS Bin 9",
    # "POPS Bin 10",
    # "POPS Bin 11"
    }
#     imet_columns2use
    
    if verbose:
        print('======')
        for var in ds.variables:
            print(var)
    
    df = pd.DataFrame()
    for key in imet_columns2use.keys():
        df[imet_columns2use[key]] = ds[key].to_pandas()
    return df

def set_altitude_column(imet, alt_source):
    if 'baro' in alt_source:
        imet['altitude'] = imet['altitude_ptu']
    elif 'gps' in alt_source:
        imet['altitude'] = imet['altitude_gps']
    elif alt_source == 'bad':
        return False
    else:
        raise ValueError('alt_source unknown: {}'.format(alt_source))

    imet.drop(['altitude_ptu', 'altitude_gps'], axis=1,  inplace=True)

    imet.altitude *= 1000
    return True

def load_met_files(start_time, end_time, folders):
    fnames = [i for i in folders['path2met_folder'].glob('*.cdf')]
    met_start = [pd.to_datetime(i.name.split('.')[2]) for i in fnames]
    met_file_df = pd.DataFrame({'path': fnames}, index = met_start)
    met_file_df.sort_index(inplace=True)
    met_file_df_match = met_file_df.truncate(start_time - pd.Timedelta(1, 'D'), end_time)

    tl = []
    for idx, path in met_file_df_match.iterrows():
        ds = xr.open_dataset(path[0])
        press = ds.atmos_pressure.to_dataframe()
        press_resamp = press.resample('1s').interpolate()
        tl.append(press_resamp)


    press_df = pd.concat(tl, sort=True)

    press_df = press_df.truncate(start_time, end_time)
    return press_df * 10

def add_eqiv_potential_temp(tbs):
    temp = tbs['temp'].values * units.celsius
    rh = tbs['rh'].values * units.percent
    press = tbs['atm_pressure'].values * units.millibar

#     dewpt_pint = calc.dewpoint_rh(temp, rh)
    dewpt_pint = calc.dewpoint_from_relative_humidity(temp, rh)
    dewpt = np.array(dewpt_pint)

    tbs['dew_point'] = dewpt
    tbs['equiv_potential_temperature'] = np.array(calc.equivalent_potential_temperature(press, temp, dewpt_pint))
    return

def add_cloud_base_distance_and_transit(dst, ceilometer_folder = '/mnt/telg/data/arm_data/OLI/ceilometer/'):
    # find ceilometer files around the tbs time
    ## get availble ceilometer files
    path = pathlib.Path(ceilometer_folder)
    df = pd.DataFrame(list(path.glob('*.nc')), columns = ['path'])
    df.index = df.path.apply(lambda pt: pd.to_datetime(' '.join(pt.name.split('.')[2:4])))
    df.index.name = 'datetime'
    df['name'] = df.path.apply(lambda x: x.name)
    df.sort_index(inplace=True)

    tdiff = abs(df.index - dst.datetime.values[0])
    tdmin, tdargmin = tdiff.min(), tdiff.argmin()
    assert(tdmin < pd.Timedelta(1, 'd')) # if this is False there is basically no ceilometer data found close by
    df_sel = df.iloc[tdargmin-1:tdargmin+2,:]
    ceil = arm.read_ceilometer_nc([pt for pt in df_sel.path])

    # distance of cloud base to flight path
    # measurement frequencies differ a lot between ceilometer and nasascience therefore I will interpolate the ceilometer data to the nsa_sceince measurment frequency

    ct = ceil.copy()
    cb = ct.cloudbase.data.resample('1s').bfill().reindex(dst.datetime.values)

    dst_alt = dst.altitude.to_pandas()
    dist2cb = (cb * (-1)).add(dst_alt, axis = 0).First_cloud_base

    # take each minimum fit it and determine the local cloud base transit

    def find_section_with_cb_trans(dist2cb, trans, window = 150):
        td = pd.to_timedelta(f'{window}s')
        dist2cb_sel = dist2cb.loc[trans - td: trans + td]
        dist2cb_sel_df = pd.DataFrame(dist2cb_sel)
        return dist2cb_sel_df

    dist2cb_tmp = dist2cb.copy()
    trans_df = pd.DataFrame(columns = ['trans'])

    # loop over all minima
    direction_min_r2 = 0.2
    while True:
        valid = True
        trans = dist2cb_tmp.abs().idxmin()
        trans

        off_center = True

        its_max = 5
        its = 0
        while off_center and its < its_max:
            its += 1
            dist2cb_sel_df = find_section_with_cb_trans(dist2cb_tmp, trans)

            dist2cb_sel_df['idx'] = np.arange(dist2cb_sel_df.shape[0])
            dist2cb_sel_df.dropna(inplace=True)
            out = sp.stats.linregress(dist2cb_sel_df.idx.values, dist2cb_sel_df.First_cloud_base.values)
            dist2cb_sel_df['linreg'] =  dist2cb_sel_df.idx.apply(lambda x: out.slope * x + out.intercept)

            zero_crossing = int(round(- out.intercept / out.slope))

            # when the zero crossing is outside the region, the minimum was not an actual crossing anymore
            if not (dist2cb_sel_df.idx.dropna().min() < zero_crossing < dist2cb_sel_df.idx.dropna().max()):
                valid = False
                break

            # in case zero crossing was nan, take the closest:
            if zero_crossing not in dist2cb_sel_df.idx:
                zero_crossing = dist2cb_sel_df.idx[abs(dist2cb_sel_df.idx  - zero_crossing).idxmin()]
            new_trans = dist2cb_sel_df[dist2cb_sel_df.idx == zero_crossing].index.values[0]

            tol = (dist2cb_sel_df.index[-1] - dist2cb_sel_df.index[0]) / 5
            off_center = pd.to_timedelta(abs(trans - new_trans)) > tol
            trans = new_trans
        if not valid:
            break

        if out.rvalue**2 < direction_min_r2:
            direction = 0
        elif out.slope < 0:
            direction = -1
        elif out.slope > 0:
            direction = 1

        trans_df = trans_df.append(pd.DataFrame([direction], columns = ['trans'], index = [trans]))
        trans_df

        dist2cb_tmp.loc[dist2cb_sel_df.index] = np.nan

    trans_df.sort_index(inplace=True)

    # add to xarray dataset
    dst['remote_ceilometer_cloud_base_altitude'] = cb.First_cloud_base
    
    trans_df = pd.DataFrame(trans_df, index = dst_alt.index)
    dst['cloud_base_transit'] = trans_df.trans.astype(float)
    dst.cloud_base_transit.attrs['info'] = f'1 - acent, -1 - decent, 0 - unclear (r^2 < {direction_min_r2})'

    dist2cb.index.name = 'datetime'
    dst['cloud_base_distance'] = dist2cb
    dst.cloud_base_distance.attrs['info'] = 'Distance to cloud: "cloude base height" - "instrument altitude" '
    dst.cloud_base_distance.attrs['unit'] = 'meters'
    return dst

def add_uhsas_stuff(dst, pops_dist, uhsas_folder = '/mnt/telg/data/arm_data/OLI/uhsas/', d_overlap = [140, 1000]):
    """ adds number concentration from the UHSAS as well as nc for the overlap
    of uhsas and pops (both ways)
    
    Ideas
    -----
    rebin the data ... interpolating would not to the trick bacause of the low counts in larger bins ... try writing my own rebinning routin"""

# merge uhsas files
## find the matching files
    path = pathlib.Path(uhsas_folder)
    df = pd.DataFrame(list(path.glob('*.nc')), columns = ['path'])
    df.index = df.path.apply(lambda pt: pd.to_datetime(' '.join(pt.name.split('.')[2:4])))
    df.index.name = 'datetime'
    df['name'] = df.path.apply(lambda x: x.name)
    df.sort_index(inplace=True)

    tdiff = abs(df.index - dst.datetime.values[0])
    tdmin, tdargmin = tdiff.min(), tdiff.argmin()

    # if tdmin is larger than 1 day there is basically no data found close by
    if tdmin > pd.Timedelta(1, 'd'): 
        raise ValueError('no UHSAS data available')

    df_sel = df.iloc[tdargmin-1:tdargmin+2,:]

## load the files
    uhsas = arm.read_uhsas([pt for pt in df_sel.path])
    udist = uhsas.copy()
    udist.data.index.name = 'datetime'
    
## resample data -- the sampling frequency of the UHSAS is 10s while ours is 1s ... lineare interpolate
    udist.data = udist.data.resample('1s').bfill().reindex(dst.datetime.values)

## determine diameter overlab with POPS --- both ways   
### cut lower end of uhsas
    udist_zoom = udist.zoom_diameter(d_overlap[0])

### cut upper end of POPS
    pops_dist_zoom = pops_dist.zoom_diameter(end = d_overlap[1])

## add data to product
    dst['pops_particle_number_concentration_uhsas_overlap'] = pops_dist_zoom.particle_number_concentration.data.iloc[:,0]
    dst['ground_uhsas_particle_number_concentration_pops_overlap'] = udist_zoom.particle_number_concentration.data.iloc[:,0]
    dst['ground_uhsas_particle_number_concentration'] = udist.particle_number_concentration.data.iloc[:,0]
    return

def add_mwr_products(dst, path = '/mnt/telg/data/arm_data/OLI/max_mwr/MWR/'):
    """
    Adds liquid water path from micro wave radar

    Parameters
    ----------
    dst : TYPE
        DESCRIPTION.
    path : TYPE, optional
        Folder where to find mwr data files. The default is '/mnt/telg/data/arm_data/OLI/max_mwr/MWR/'.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    path = pathlib.Path(path)

    # merge micro wave radar data
    ## find the matching files
    df = pd.DataFrame(list(path.glob('*.cdf')), columns = ['path'])

    df.index = df.path.apply(lambda pt: pd.to_datetime(' '.join(pt.name.split('.')[2:4])))
    df.index.name = 'datetime'
    df['name'] = df.path.apply(lambda x: x.name)
    df.sort_index(inplace=True)

    tdiff = abs(df.index - dst.datetime.values[0])
    tdmin, tdargmin = tdiff.min(), tdiff.argmin()

    # if tdmin is larger than 1 day there is basically no data found close by
    if tdmin > pd.Timedelta(1, 'd'): 
        raise ValueError('no MWR data available')

    df_sel = df.iloc[tdargmin-1:tdargmin+2,:]

    ## open datasets and merge relevant data
    lwp_list = []
    for pt in df_sel.path:
        ds_mwr = xr.open_dataset(pt)
        date_time = ds_mwr.time_offset.to_pandas()
        lwp = ds_mwr.phys_lwp.to_pandas()
        lwp.index = date_time
        lwp.index.name = 'datetime'
        lwp_list.append(lwp)
    lwp = pd.concat(lwp_list)

    ## resample data -- the sampling frequency of the UHSAS is 10s while ours is 1s ... lineare interpolate
    lwp = lwp.resample('1s').bfill().reindex(dst.datetime.values)

    ## add to nsascience dataset
    dst['remote_mwr_liquid_water_path'] = lwp.astype(float)
    
    return

def add_cloud_top(dst, path = '/mnt/telg/data/arm_data/OLI/max_cloudtop/'):
    path = pathlib.Path(path)

    ## find the matching files
        #make sure the start and end of dst is in the same month, then look for the right cloud top file
    df = pd.DataFrame(list(path.glob('*.nc')), columns = ['path'])
    fct = lambda pt: pd.to_datetime(pt.name.split('.')[0].split('_')[-1]+'01')
    df.index = df.path.apply(fct)
    # df

    dst_start = pd.to_datetime(dst.datetime.values[0])
    dst_end = pd.to_datetime(dst.datetime.values[-1])

    if dst_start.month != dst_end.month:
        raise ValueError('start and end of nsasience product is not in the same month ... programming required')
        
    df['match_year'] = df.apply(lambda x: x.index.year == dst_start.year)
    df[['match_month','no_idea']] = df.apply(lambda x: x.index.month == dst_start.month)
    path_ct = df[np.logical_and(df.match_month, df.match_year)].path.iloc[0]

    ## open the file
    ct_ds = xr.open_dataset(path_ct)
    ct_df = ct_ds.robustCloudTop.to_pandas()

    ## resample data -- the sampling frequency of the the cloud top product is 15s while ours is 1s ... backfill ... lineare interpolate only works if the only nans are the newly introduced
    cltop = ct_df.resample('1s').bfill().reindex(dst.datetime.values)

    # merge into nsascience product
    cltop.index.name = 'datetime'
    dst['remote_kazr_cloud_top'] = cltop
    
    return

def add_sectioning(dst,
                   folders=False,
                   path2product_file = None,
                   pen = 5,
                   threshold_ground_alt = 50,
                   ignor_altitudes_smaller_than = 0,
                   threshold_park_slope = 0.03,
                   has2be_connected2ground = True,
                   mintime2interrupt_profile ='10m',
                   mintimedelta='2m',
                   minprofile_altdiff = 20,
                   minprofile_duration = '10m'):
    """
    Parameters
    -----------
    folders: if dict then a figur is generated and saved in quicklooks"""

    def plot_profiles(out, save2path, path2product_file = None):
        sectioned_detailed = out['sectioned_deteiled']
        sectioned_coarse = out['sectioned_coarse']
        f, aa = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
        a_detail, a_coarse = aa
        alt = dst.altitude.to_dataframe()
        for a in aa:
            alt.plot(ax=a, lw=1, color='black')

        for prof in sectioned_detailed.profile.dropna().unique():
            alt[sectioned_detailed.profile == prof].plot(ax=a_detail, lw=2)

        for prof in sectioned_coarse.profile.dropna().unique():
            alt[sectioned_coarse.profile == prof].plot(ax=a_coarse, lw=2)

        # title
        # dt = pd.to_datetime(dst.datetime.values[0]).__str__().replace('-', '').replace(':', '').replace(' ', '.')
        # popssn = dst.based_on_file_pops.split('.')[-1]
        # title = f'{dt}.pops{popssn}'
        title = path2product_file.name.replace('.nc','')
        a_detail.set_title(title)

        a_coarse.set_xlabel('')

        label = ['detailed', 'coarse']
        for e, a in enumerate(aa):
            a.set_ylim(ymin=-20)
            leg = a.legend()
            leg.remove()
            a.text(0.85, 0.85, label[e], transform=a.transAxes,
                   bbox=dict(boxstyle='round', facecolor=[1, 1, 1], alpha=0.5))

        plt_tools.axes.labels.set_shared_label(aa, 'Altitude (m)', axis='y', )
        if not isinstance(save2path, type(None)):
            f.patch.set_alpha(0)
            pathout = save2path.joinpath('sectioning.' + title + '.png')
            f.savefig(pathout, bbox_inches='tight')
        return f, aa

    def destill_profiles(dst,
                         #                      profile2top = False
                         ):
        # precondition the data

        def profile2top(sectioned):
            sect_tmp = sectioned.copy()
            sect_cponly = sect_tmp[~sect_tmp.changepoints.isna()]
            # sect_tmp.classfied[~sect_tmp.changepoints.isna()] == 0
            # sect_cponly.classfied == 0
            profcounter = 1
            for ts, row in sect_cponly.iterrows():
                if row.profile == -8888:
                    profcounter += 1
                elif np.isnan(row.profile):
                    continue
                else:
                    sect_tmp.loc[ts, 'profile'] = profcounter
            return sect_tmp

        def fillit(sectioned_tmp):
            sectioned_tmp.profile.ffill(inplace=True)
            sectioned_tmp.profile[sectioned_tmp.profile == 0] = np.nan
            sectioned_tmp.profile[sectioned_tmp.profile == -9999] = np.nan
            sectioned_tmp.profile[sectioned_tmp.profile == -8888] = np.nan

        def remove_cp_when_section2short(df, mintimedelta ):
            df[~df.changepoints.isna()]
            idx = df[~df.changepoints.isna()].index
            idx_delta = idx[1:] - idx[:-1]
            mintimedelta = pd.to_timedelta(mintimedelta)
            where = idx_delta < mintimedelta
            where = np.append(where[::-1], False)[::-1]
            df.loc[idx[where], 'changepoints'] = np.nan

        alt = dst.altitude.to_dataframe().altitude

    ## remove falty datapoints
        alt[alt < ignor_altitudes_smaller_than] = np.nan

    ## smoothen the altitude and then get the derivative
        ints = 5

        altsmooth = alt.resample(f'{ints}s').mean()
        div = np.gradient(altsmooth)
        altsmooth = pd.DataFrame(altsmooth)
        altsmooth['deriv'] = div

        if 0:
            a = alt.plot()
            altsmooth.altitude.plot(ax=a)

            at = a.twinx()
            altsmooth.deriv.plot(ax=at, color=colors[2])

    # ruptures

        ## make the points for the change point analysis
        altsmooth.interpolate(inplace=True)
        altsmooth.dropna(inplace=True)
        points = altsmooth.deriv.values

    ## do the change point analysis
        model = "rbf"
        algo = rpt.Pelt(model=model).fit(points)
        result = algo.predict(pen=pen)

    ## plot results
        # %matplotlib inline
        # plt.rcParams['figure.dpi']=200

        if 0:
            f, aa = rpt.display(points, result, figsize=(10, 6))
            for a in aa:
                at = a.twinx()
                at.plot(altsmooth.altitude.values, color=colors[1])

    # convert the change point info into profiles
        sectioned = pd.DataFrame(index=dst.datetime.to_pandas().index,
                                 columns=['changepoints', 'slope', 'classfied', '_alt_avg', '_alt_at_cp', '_alt_min','profile'])

    ## mark change points
        sectioned.changepoints.loc[altsmooth.iloc[result[:-1]].index] = 1
        sectioned.changepoints.iloc[-1] = 1

    ## remove change points where section is too short
        remove_cp_when_section2short(sectioned, mintimedelta)

    ## add aditional parameters to sections between change points
        startime = sectioned.index[0]
        for ts, cp in sectioned.changepoints.dropna().iteritems():
            endtime = ts

            sect = alt.loc[startime:endtime]
            df = pd.DataFrame(sect)
            df['s'] = range(df.shape[0])
            dfwn = df.copy()  # a copy with all the nans still in it
            df.dropna(inplace=True)

            if df.shape[0] == 0:
                continue
            res = sp.stats.linregress(df.s, df.altitude)
            sectioned.slope.loc[dfwn.index] = res.slope
            sectioned._alt_avg.loc[dfwn.index] = sect.mean()
            sectioned._alt_min.loc[dfwn.index] = sect.min()
            sectioned._alt_at_cp.loc[ts] = sect.dropna().iloc[-1]
            startime = ts

    ## classify sections between change points
        ground = 0
        park = 1
        up = 2
        down = 3
        sectioned.classfied[sectioned.slope.abs() < threshold_park_slope] = park
        sectioned.classfied[sectioned.slope >= threshold_park_slope] = up
        sectioned.classfied[sectioned.slope < -threshold_park_slope] = down

        where = np.logical_and(sectioned.classfied == park, sectioned._alt_min.interpolate() < threshold_ground_alt)
        sectioned.classfied[where] = ground

    ## combine sections into profiles
        profile_id = 0
        cp1 = 0
        cp2 = 0
        cp3 = 0
        uod = lambda x: x if row.classfied == 2 else -x
        # profile_id = iter(range(100))
        inside = False

        sect_cponly = sectioned[~sectioned.changepoints.isna()]
        for ts, row in sect_cponly.iterrows():
            # for ts, row in sectioned.iterrows():
            try:
                #         print('works')
                row_next = sect_cponly.iloc[sect_cponly.index.get_loc(ts) + 1]
                # ts_next = row_next.name
            except IndexError:
                #         print('doese')
                # this happens if we are at the end of the list
                row_next = row.copy()
                row_next.name = pd.to_datetime('2200-01-01 00:00:00')
                row_next.classfied = -9999

            try:
                row_next_next = sect_cponly.iloc[sect_cponly.index.get_loc(row_next.name) + 1]
                # ts_next_next = row_next_next.name
            except:
                row_next_next = row.copy()
                row_next_next.name = pd.to_datetime('2200-01-01 00:00:00')
                row_next_next.classfied = -9999
                # row_next_next = False
                # ts_next_next = pd.to_timedelta('2200-01-01 00:00:00')

            #     if ts.__str__() == '2017-05-23 17:55:45':
            #         break
            if not inside:
                if row.classfied == ground:
                    continue
                elif row.classfied == park:
                    continue
                elif pd.isna(row.classfied):
                    continue
                elif row.classfied in [up, down]:
                    #                 cp1 += 1
                    profile_id += 1
                    sectioned.loc[ts, 'profile'] = uod(profile_id)
                    inside = row.classfied
                    time_at_classified = ts
                else:
                    assert (False)  # should not be possible
            if inside in [up, down]:
                if row.classfied == ground:
                    sectioned.loc[ts, 'profile'] = -8888
                    inside = False
                elif row.classfied == inside:
                    sectioned.loc[ts, 'profile'] = uod(profile_id)
                    time_at_classified = ts # This is the time at which a valid change is happening

                # elif not isinstance(row_next_next, bool) :
                elif ((row_next.name - time_at_classified) < pd.to_timedelta(mintime2interrupt_profile)) and row_next_next.classfied == inside:
                    # if not isinstance(row_next_next, bool):
                    # if row_next_next.classfied == inside:
                    sectioned.loc[ts, 'profile'] = np.nan

                elif row.classfied == park:
                    # if row_next.classfied == inside:
                    #     sectioned.loc[ts, 'profile'] = np.nan  # uod(profile_id)
                    # if (row_next.name - time_at_classified) < pd.to_timedelta(mintime2interrupt_profile):
                    #     sectioned.loc[ts, 'profile'] = np.nan
                    # else:
                    sectioned.loc[ts, 'profile'] = 0
                    inside = False
                #             continue

                elif pd.isna(row.classfied):
                    #                 cp2 += 1
                    inside = False
                    sectioned.loc[ts, 'profile'] = -9999
                    continue
                elif row.classfied != inside:
                    #                 cp3 += 1
                    # if (row_next.name - time_at_classified) < pd.to_timedelta(mintime2interrupt_profile):
                    #     sectioned.loc[ts, 'profile'] = np.nan
                    # else:
                    profile_id += 1
                    sectioned.loc[ts, 'profile'] = uod(profile_id)
                    inside = row.classfied
                else:
                    assert (False)  # noep
            else:
                assert (False)  # should not be possible

    ## combine even further if profiles comprise of everything up to the top and then down again

        sectioned_coarse = profile2top(sectioned)


    ## fill all the gaps inbetween the checkpoints

        fillit(sectioned)
        fillit(sectioned_coarse)

        ## continueation of global profiles (s.o.) distinguish between up and down
        for profcounter in sectioned_coarse.profile.dropna().unique():
            altsec = alt[sectioned_coarse.profile == profcounter]
            tmax = altsec.idxmax()
            sectioned_coarse.profile[
                np.logical_and(sectioned_coarse.profile == profcounter, sectioned_coarse.index > tmax)] = -profcounter

        #     break

    ## test if profile reaches the ground and remove if not
        def test_if_grounded(sectioned_tmp):
            minaltsofprofs = alt.groupby(sectioned_tmp.profile).min()

            for prof, minalt in minaltsofprofs[minaltsofprofs > threshold_ground_alt].iteritems():
                sectioned_tmp.profile[sectioned_tmp.profile == prof] = np.nan
        if has2be_connected2ground:
            test_if_grounded(sectioned)
            test_if_grounded(sectioned_coarse)

    ## remove sections that are too short or not enought elevation difference
        def remove_profiles_2short_or_heigh(sectioned):
            for pf in sectioned.profile.dropna().unique():
                altsect = alt[sectioned.profile == pf]
                dt = altsect.index[-1] - altsect.index[0]
                dalt = abs(altsect.max() - altsect.min())
                if (dt < pd.to_timedelta(minprofile_duration)) or dalt < minprofile_altdiff:
                    sectioned.profile[sectioned.profile == pf] = np.nan

        remove_profiles_2short_or_heigh(sectioned)
        remove_profiles_2short_or_heigh(sectioned_coarse)

    ## if one was deleted we have to shift all sucessive ones to avoid confusion
        while 1:
            profnos = abs(sectioned.profile.dropna().unique())
            diff = profnos[1:] - profnos[:-1]
            if len(np.unique(diff)) > 1:
                idx = diff.argmax() + 1
            else:
                break

            for pn in profnos[idx:]:
                where = sectioned.profile.abs() == pn

                if sectioned.profile[where].unique()[0] > 0:
                    fct = 1
                elif sectioned.profile[where].unique()[0] < 0:
                    fct = -1
                sectioned.profile[where] = fct * (pn - 1)


        ## plot result
        #     if 1:
        #         # %matplotlib inline
        #         # plt.rcParams['figure.dpi']=200

        #         a = sectioned.profile.plot()
        #         at = a.twinx()
        #         alt.plot(ax = at, color = colors[1])
        #         for idx in sect_cponly.index:
        #             a.axvline(idx, color = colors[2], lw = 0.5, ls = '--')

        out = {}
        out['sectioned_deteiled'] = sectioned
        out['sectioned_coarse'] = sectioned_coarse
        return out

    out = destill_profiles(dst)

    sectioned_detailed = out['sectioned_deteiled']
    sectioned_coarse = out['sectioned_coarse']

    dst['sectioned_deteiled'] = sectioned_detailed
    dst['sectioned_coarse'] = sectioned_coarse
    # dst.sectioned_deteiled.attrs['info'] = ''
    info = 'This variable is one of two tables that describe ways to split up the flight path into sections that constitude verticle profiles. The "coarse" version of the two seperates different profiles merely by the fact that the path touched the ground and up and down are seperated by the top of the profile. The "deteiled" version only considers sections as profiles when they are not seperated by an intermittened downturn. It is also required for a profile to reach the ground. In the futhure we could consider limiting this even further by not allowing park position longer then xy to be present in a profile.'
    dst.sectioned_coarse.attrs['info'] = info
    dst.sectioned_deteiled.attrs['info'] = info

    if folders:
        plot_profiles(out, folders['path2quicklooks'], path2product_file = path2product_file)

def process(ipmatchrow, folders, test = False,raise_error = True, path2product_file = None):
    # imet
    ####
    imet = open_iMet(folders['path2imet_folder'].joinpath(ipmatchrow.fn_imet))
    didit = set_altitude_column(imet, ipmatchrow.which_alt)

    if not didit:
        return False

    ## fill missing timestamps with nans
    imet = imet.resample('1s').mean()

    start_time, end_time = imet.index.min(),imet.index.max()

    # POPS
    ds = xr.open_dataset(folders['path2pops_folder'].joinpath(ipmatchrow.fn_pops + '.nc'))

    ## size distribution
    dist = ds.size_distributions.to_pandas()
    dist = dist.resample('1s').mean()
    dist = dist.truncate(start_time, end_time)

    dist_ts = sd.SizeDist_TS(dist, size_distribution.diameter_binning.bincenters2binsANDnames(ds.bincenters.values)[0], 'dNdlogDp')
    dist_ts._data_period = 1

    particle_no_concentration = dist_ts.particle_number_concentration.data.copy()

    particle_mean_d = dist_ts.particle_mean_diameter.data.copy()

    ## housekeeping
    df = ds.housekeeping.to_pandas()
    df = df.Altitude
    df = df.resample('1s').mean()
    df = df.truncate(start_time, end_time)

    # merge
    tbs = imet.copy()
    tbs['pops_particle_number_concentration'] = particle_no_concentration
    tbs['pops_particle_mean_diameter'] = particle_mean_d
    tbs['test_POPS_altitude'] = df.copy()


# met
## ground pressure
    tbs['atm_pressure_ground'] = load_met_files(start_time, end_time, folders)

# retrievals
## potential temperatur
    add_eqiv_potential_temp(tbs)

# create xarray dataset
    dstbs = xr.Dataset(tbs)
    dstbs['pops_size_distribution'] = dist

# additional retrievals that take the xarray dataset to work with
    try:
        add_cloud_base_distance_and_transit(dstbs)
    except:
        if raise_error:
            print('Error in adding adding cloud base stuff: ', sys.exc_info())
        else:
            return dstbs
    # add_sectioning(dstbs, folders=folders, path2product_file = path2product_file)
    add_uhsas_stuff(dstbs, dist_ts, uhsas_folder = folders['path2uhsas'])
    try:
        add_mwr_products(dstbs, folders['path2mwr'])
    except Exception as e:
        txt = e.__str__()
        warnings.warn(txt)
    add_cloud_top(dstbs, folders['path2cloudtop'])
    
    if test:
        out = {}
        out['tbs'] = tbs
        out['start'] = start_time
        out['end'] = end_time
        return out
    else:
        return dstbs

#####
# database stuff
#####

def get_pops2imet_matching_table(path2database):
    with sqlite3.connect(path2database) as db:

        tbl_name = 'match_datasets_imet_pops'
        qu = 'select * from {}'.format(tbl_name)
        imet_pops_match = pd.read_sql(qu, db)
    return imet_pops_match

def make_product(version = '',
                 changelog = '',
                 folders = {},
                 test = False):

    if not folders['path2product'].exists():
        folders['path2product'].mkdir()
        path2quicklooks = pathlib.Path(folders['path2product']).joinpath('quicklooks')
        path2quicklooks.mkdir()
        folders['path2quicklooks'] = path2quicklooks
        print('generated product folder')

    fails = []

    imet_pops_match = get_pops2imet_matching_table(folders['path2database'])
    for idx, ipmatchrow in imet_pops_match.iterrows():
        # generate tbs name

    #     dt = pd.to_datetime(flt.start)
    #     start = '{:04d}{:02d}{:02d}.{:02d}{:02d}{:02d}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    #     fname_tbs = 'olitbs.{start}.{version:03d}.nc'.format(start=start, version = version)
    #     path2tbs = path2tbs_folder.joinpath(fname_tbs)

    #     print(fname_tbs, end = ' ... ')

        st = '.'.join(ipmatchrow.fn_imet.split('.')[1:3])
        path2product_file  = folders['path2product'].joinpath('{}.{}.popssn{}{}'.format(folders['path2product'].name, st, ipmatchrow.popssn,'.nc'))
        print(path2product_file, end = ' ... ')

        if path2product_file.exists():
            print('file exists -> skip')
            continue
    #     ds = process(ipmatchrow)
    #     break
        try:
            ds = process(ipmatchrow, folders, path2product_file = path2product_file)
            if not ds:
                txt = 'ds is not (execution of process(ipmatchrow) failed)'
                fails.append({'id': idx, 'cause': txt})
                print(txt)
                continue
            ds.attrs['altitude_source_quality'] = ipmatchrow.which_alt
            ds.attrs['based_on_file_imet'] = ipmatchrow.fn_imet
            ds.attrs['based_on_file_pops'] = ipmatchrow.fn_pops
            ds.attrs['version'] = version
            ds.attrs['changelog'] = changelog
        except FileNotFoundError:
            txt = 'file not found ... required file missing'
            fails.append({'id': idx, 'cause': txt})
            print(txt)
            continue
        except Exception as e:
            txt = f'{e.__str__()}'
            fails.append({'id': idx, 'cause': txt, 'ipmatchrow':ipmatchrow})
            print(txt)
            if test:
                return ipmatchrow
            continue


    #     if not ds:
    #         print('no ds -> skip')
    #         continue

        ds.to_netcdf(path2product_file)
        print('done')
        if test:
            break
    return fails