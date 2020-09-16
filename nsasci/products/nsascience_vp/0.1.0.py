# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import xarray as xr
from atmPy.general import timeseries as ts
import numpy as np
from nsasci.products.nsascience_vp import quicklooks


def testing(group):
    # every flight has launch and landing
    if ((group.type == 'launch').sum() != (group.type == 'landing').sum()):
        raise ValueError(f'launch != landing for {group.file_name}')
    
    
    # every flight has a top
    if ((group.type == 'launch').sum() != (group.type == 'top').sum()):
        raise ValueError(f'launch != top for {group.file_name}')
    return True
    
# testing(group)

def add_df2ds(df, ds, addon):
    for var in df:
        vnds = f'{var}_{addon}'
        ds[vnds] = df[var]
    return

def get_changepoints_from_db(path2db):
    # read database
    tbl_name = 'change_points'
    with sqlite3.connect(path2db) as db:
        qu = 'select * from "{}"'.format(tbl_name)
        qu_out = pd.read_sql(qu, db)

    # condition the dataframe
    df_cp = qu_out.copy()
    df_cp.index = df_cp.datetime
    df_cp.sort_index(inplace=True)
    return df_cp

def process(path2inputfile, productinfo, changepoints, nsascience_fn, path2outputfld = None, save2netcdf = True, save_quicklook = True):
    fn = nsascience_fn
    df_cp = changepoints
    ds = xr.open_dataset(path2inputfile)
    productinfo['input.nsascience.version'] = ds.attrs['version']
    productinfo['input.nsascience.file_name'] = path2inputfile.name
    # get values that are turned into vertical profile
    ignorelist = ['datetime', 'test_POPS_altitude', 'bincenters', 'pops_size_distribution',  'cloud_base_distance']
    df_nsa = pd.DataFrame()
    for var in ds.variables:
        if var in ignorelist:
            continue
        else:
            df_nsa[var] = ds[var].to_pandas()        

    # group changepoints
    group = df_cp[df_cp.file_name == fn]

    # get sets of launch, top, landing

    launches = group[group.type == 'launch'].datetime.values
    tops = group[group.type == 'top'].datetime.values
    landings = group[group.type == 'landing'].datetime.values

    for launch, top, landing in zip(launches, tops, landings):
    #     break

        ## generate output path ... this has to happen here
        popssn = fn.split('.')[-2]
        dtt = pd.to_datetime(launch)
        fn_out = f'nsascience_vertical_profile.{dtt.year}{dtt.month:02d}{dtt.day:02d}.{dtt.hour:02d}{dtt.minute:02d}{dtt.second:02d}.{popssn}.nc'
        path_out = path2outputfld.joinpath(fn_out)

        print(f'next up: {path_out}')
        if path_out.is_file():
            txt = 'file exists ... skip'
            print('\t' + txt)
            continue
    #         raise FileExistsError(txt)

        # get profile up and down
        df_nsa_up = df_nsa.truncate(launch,top)
        df_nsa_down = df_nsa.truncate(top, landing)

        # get all the relevant avg values
        avg_list = ['ground_atm_pressure', 'remote_ceilometer_cloud_base_altitude', 
#                     'remote_mwr_liquid_water_path', 
                    'remote_kazr_cloud_top']
        avg_list_res = []
        for avg in avg_list:
        #     break
            up = df_nsa_up.loc[:,avg]
            down = df_nsa_down.loc[:,avg]
            out = dict(name = avg,
                       up = up.mean(),
                       up_std = up.std(),
                       down = down.mean(),
                       down_std = down.std())
            avg_list_res.append(out)

        ## remove the average values
        df_nsa_up= df_nsa_up.drop(avg_list, axis=1)
        df_nsa_down = df_nsa_down.drop(avg_list, axis=1)

        # turn timeseries into vertical profile

        ts_nsa_up = ts.TimeSeries(df_nsa_up)
        ts_nsa_down = ts.TimeSeries(df_nsa_down)

        resolution = 10

        top = max([ts_nsa_up.data.altitude.max(), ts_nsa_down.data.altitude.max()])
        top = np.ceil(top / resolution) * resolution

        bottom = min([ts_nsa_up.data.altitude.min(), ts_nsa_down.data.altitude.min()])
        bottom = np.floor(bottom/resolution) * resolution
        if bottom > 0:
            bottom = 0

        resolution = (resolution, bottom, top)
        vp_nsa_up, vp_nsa_up_std = ts_nsa_up.convert2verticalprofile(altitude_column='altitude', resolution=resolution, return_std=True)
        vp_nsa_down, vp_nsa_down_std = ts_nsa_down.convert2verticalprofile(altitude_column='altitude', resolution=resolution, return_std=True)
        # resolution

        ## the dataset
        ds_out = xr.Dataset()

        ## prepare 1d variables

        vp_nsa_up.data.index.name = 'altitude'
        vp_nsa_up_std.data.index.name = 'altitude'
        vp_nsa_down.data.index.name = 'altitude'
        vp_nsa_down_std.data.index.name = 'altitude'

#         return vp_nsa_up, vp_nsa_up_std, vp_nsa_down, vp_nsa_down_std, ds_out
        ## add 1d variables to dataset
    
        ignore_list = ['altitude', 'cloud_base_transit', 'DateTime']
#         return vp_nsa_up.data
        for var in vp_nsa_up.data:
            if var in ignore_list:
                continue

            # var = 'pops_particle_mean_diameter'

            df = pd.DataFrame()
            df['up'] = vp_nsa_up.data[var]
            df['up_std'] = vp_nsa_up_std.data[var]
            df['down'] = vp_nsa_down.data[var]
            df['down_std'] = vp_nsa_down_std.data[var]
            ds_out[var] = df
            
        # sort variables in the file ... could not find a attribute that does that?!?
        varlist = list(ds_out.variables)
        varlist.sort()

        # remove some artefacts
#         varlist = [var for var in varlist if 'DateTime' not in var]
#         varlist = [var for var in varlist if 'altitude' not in var]

        # regenerate the dataset sorted and cleaned
        dst = xr.Dataset()
        for var in varlist:
            dst[var] = ds_out[var]

        ds_out = dst

        # add the average remote values
        avg_list_res.sort(key=lambda x: x['name'])
#         return avg_list_res, ds_out
        for avg in avg_list_res:
#             avg = avg.copy()
            name = avg.pop('name')
            st =  pd.Series(avg)
            st.index.name = 'dim_1'
            ds_out[name] = st
        
    
#         for avg in avg_list_res:
#             name = avg['name']
#             keys = [i for i in list(avg.keys()) if i != 'name']
#             keys.sort()
#             for k in keys:
#                 ds_out[f'{name}_{k}'] = avg[k]
                
        # add productinfo
        
        for key in productinfo:
            ds_out.attrs[key] = productinfo[key]

        # save to netcdf
        if save2netcdf:
            ds_out.to_netcdf(path_out)
        
        if save_quicklook:
            f, aa = quicklooks.plot_quicklook(ds_out, plot_cb = True, plot_ct = True, save=True, output_path = path_out)
            f, aa = quicklooks.plot_quicklook(ds_out, plot_cb = True, plot_ct = False, save=True, output_path = path_out)
            f, aa = quicklooks.plot_quicklook(ds_out, plot_cb = False, plot_ct = False, save=True, output_path = path_out)
    
    out = {}
    try:
        out['ds_nsascience'] = ds
        out['output_path'] = path_out
        out['product_info'] = productinfo
        out['ds_out'] = ds_out
    except:
        pass
    return out