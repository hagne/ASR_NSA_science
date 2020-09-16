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
    trans_df = pd.DataFrame(trans_df, index = dst_alt.index)
    dst['cloud_base_transit'] = trans_df.trans
    dst.cloud_base_transit.attrs['info'] = f'1 - acent, -1 - decent, 0 - unclear (r^2 < {direction_min_r2})'

    dist2cb.index.name = 'datetime'
    dst['cloud_base_distance'] = dist2cb
    dst.cloud_base_distance.attrs['info'] = 'Distance to cloud: "cloude base height" - "instrument altitude" '
    dst.cloud_base_distance.attrs['unit'] = 'meters'
    return dst

def process(ipmatchrow, folders, test = False,raise_error = True):
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
            ds = process(ipmatchrow, folders)
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
        except:
            txt = 'unknown problem encountered -> skip'
            fails.append({'id': idx, 'cause': txt})
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