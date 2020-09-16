# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import xarray as xr
from atmPy.general import timeseries as ts
import numpy as np
from nsasci.products.nsascience_vp import quicklooks
import copy

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

def get_adjusted_top_up(df_nsa_up, group, launch, top):
    """Flight paths patterns are often comples. Here we try to get
    a profile that gets the closesed to being connected to the ground.
    Get the time series that ends when the balloon goes down the first 
    time. If hte section before that is a parking position it gets
    removed too.

    Parameters
    ----------
    df_nsa_up : TYPE
        DESCRIPTION.
    group : TYPE
        DESCRIPTION.
    launch : TYPE
        DESCRIPTION.
    top : TYPE
        DESCRIPTION.
    landing : TYPE
        DESCRIPTION.

    Returns
    -------
    top_adjusted : TYPE
        DESCRIPTION.

    """
    cbt = df_nsa_up.cloud_base_transit.dropna()
    # (todo) good weather
    if len(cbt) == 0:
        return (False, None,'up has no ground cloud base connection')
    
    # first cloud base transit
    fcbt = cbt.index[0]
    fcbt_v = cbt.iloc[0]
    assert(fcbt_v >=0)
    # (todo) how many changepoints before first cloud base transit
    group_ground2fcbt =group.truncate(launch, fcbt.__str__())
    
    # (todo) make sure there are no direction changes below cloud base ... implement if it shows up
    # assert((group_ground2fcbt.type == 'descent').sum() == 0)
    if (group_ground2fcbt.type == 'descent').sum() != 0:
        txt = 'up has direction changes below cloud base'
        return (False,None, txt)
    # first direction change after first cloud base
    group_fcbt2top = group.truncate(fcbt.__str__(), top)
    
    if (group_fcbt2top.type == 'descent').sum() > 0:
    #     top_before_direction_change = group_fcbt2top[group_fcbt2top.type == 'descent'].iloc[0].datetime
        dt = group_fcbt2top[group_fcbt2top.type == 'descent'].iloc[0].datetime
    
        idx = group_fcbt2top.index.get_loc(dt)
        if group_fcbt2top.iloc[idx - 1].type == 'park':
            dt = group_fcbt2top.iloc[idx - 1].datetime
        top_adjusted = dt
    else:
        top_adjusted = top
    return (True, top_adjusted, '')

def get_adjusted_top_down(df_nsa_down, group, top, landing):
    """Flight paths patterns are often comples. Here we try to get
    a profile that gets the closesed to being connected to the ground.
    Get the time series that ends when the balloon goes down the first 
    time. If hte section before that is a parking position it gets
    removed too.

    Parameters
    ----------
    df_nsa_up : TYPE
        DESCRIPTION.
    group : TYPE
        DESCRIPTION.
    top : TYPE
        DESCRIPTION.
    landing : TYPE
        DESCRIPTION.

    Returns
    -------
    top_adjusted : TYPE
        DESCRIPTION.

    """
    messages = []
    cbt = df_nsa_down.cloud_base_transit.dropna()
    # (todo) good weather
    # assert(len(cbt) > 0)
    if len(cbt) == 0:
        return (False, None, 'down has no ground cloud base connection')
    
    # last cloud base transit
    lcbt = cbt.index[-1]
    lcbt_v = cbt.iloc[0]
    # is the cloud pass the right direction?
    # assert(lcbt_v <= 0)
    if lcbt_v > 0:
        txt = 'cloud pass goes wrong way'
        return (False, None, txt)
    
    # (todo) how many changepoints before first cloud base transit
    group_lcbt2ground =group.truncate(lcbt.__str__(), landing)
    
    # (todo) make sure there are no direction changes below cloud base ... implement if it shows up
    # assert((group_lcbt2ground.type == 'ascent').sum() == 0)
    if (group_lcbt2ground.type == 'ascent').sum() != 0:
        txt = 'down has direction changes below cloud base'
        return (False,None, txt)
    
    # first direction change before last cloud base
    group_top2lcbt = group.truncate(top, lcbt.__str__())
        
    if (group_top2lcbt.type == 'ascent').sum() > 0:
        dt = group_top2lcbt[group_top2lcbt.type == 'ascent'].iloc[-1].datetime
    
        idx = group_top2lcbt.index.get_loc(dt)
        i = 0
    
        while 1:
            i += 1
            if (top_adjusted := group_top2lcbt.iloc[idx + i]).type == 'descent':
                break
        top_adjusted = top_adjusted.datetime
    else:
        top_adjusted = top
    return (True, top_adjusted, '\n'.join(messages))

def process(path2inputfile, productinfo, changepoints, nsascience_fn, 
            path2outputfld = None, save2netcdf = True, make_quicklooks = True,
            save_quicklook = True, show_quicklooks = True,
            test = False, verbose = False):
    """
    

    Parameters
    ----------
    path2inputfile : TYPE
        DESCRIPTION.
    productinfo : TYPE
        DESCRIPTION.
    changepoints : TYPE
        DESCRIPTION.
    nsascience_fn : TYPE
        DESCRIPTION.
    path2outputfld : TYPE, optional
        DESCRIPTION. The default is None.
    save2netcdf : TYPE, optional
        DESCRIPTION. The default is True.
    quicklooks : TYPE, optional
        DESCRIPTION. The default is True.
    save_quicklook : TYPE, optional
        DESCRIPTION. The default is True.
    test : bool, optional
        only processes one flight per file. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    def send_message(txt):
        if verbose:
            print(txt)
            
    out_return = {}

    fn = nsascience_fn
    df_cp = changepoints
    ds = xr.open_dataset(path2inputfile)
    productinfo['input.nsascience.version'] = ds.attrs['version']
    productinfo['input.nsascience.file_name'] = path2inputfile.name
    
### get 1D values that are turned into vertical profile
    ignorelist = ['datetime', 'time', 'time_offset','base_time','test_POPS_altitude', 'bincenters', 'pops_size_distribution',  'cloud_base_distance']
    df_nsa = pd.DataFrame()
    for var in ds.variables:
        if var in ignorelist:
            continue
        else:
            df_nsa[var] = ds[var].to_pandas()       
    
    out_return['df_nsa'] = df_nsa.copy()
### add cloud base transit altitude .... (future) replace with equivalent in nsascience product
    df_nsa_tmp = df_nsa.loc[:,['altitude', 'cloud_base_transit']].copy()
    df_nsa_tmp.altitude.interpolate(inplace=True)
    df_cbta = df_nsa_tmp[~df_nsa_tmp.cloud_base_transit.isna()]
    # print(df_cbta)
    df_nsa['cloud_base_transit_altitude'] = df_cbta.altitude


    
### group changepoints fore each nsascience file
    group = df_cp[df_cp.file_name == fn]
    # out_return['group'] = group.copy()
    
### add cloud base transit times to interesting times    
    df_cbta['type'] = 'cloud_base_transit'
    group = group.append(df_cbta.loc[:,['type']])
    group.index = group.index.astype(str)
    
    out_return['df_cbta'] = df_cbta.copy()
    out_return['group'] = group.copy()

    group.sort_index(inplace=True)
    
    # get sets of launch, top, landing

    launches = group[group.type == 'launch'].datetime.values
    tops = group[group.type == 'top'].datetime.values
    landings = group[group.type == 'landing'].datetime.values

### loop over flights (launch, top, landing)
    flid = 0
    for launch, top, landing in zip(launches, tops, landings):
        subgroup = group.truncate(launch,landing).loc[:,'type'].copy()
        subgroup.index = pd.to_datetime(subgroup.index)
        flid += 1
        send_message(f'flight no {flid}')
        out_return['flight'] = (launch, top, landing)
        
### generate output path ... this has to happen here
        popssn = fn.split('.')[-2]
        dtt = pd.to_datetime(launch)
        fn_out = f'nsascience_vertical_profile.{dtt.year}{dtt.month:02d}{dtt.day:02d}.{dtt.hour:02d}{dtt.minute:02d}{dtt.second:02d}.{popssn}.nc'
        path_out = path2outputfld.joinpath(fn_out)

        print(f'next up: {path_out}')
        if path_out.is_file():
            txt = 'file exists ... skip'
            send_message('\t' + txt)
            continue

### get profile up and down
        df_nsa_up = df_nsa.truncate(launch,top)
        df_nsa_down = df_nsa.truncate(top, landing)
        out_return['df_nsa_up'] = df_nsa_up.copy()
        out_return['df_nsa_down'] = df_nsa_down.copy()
        


# # get all the relevant avg values
#         avg_list = ['ground_atm_pressure', 'remote_ceilometer_cloud_base_altitude', 'cloud_base_transit_altitude',
# #                     'remote_mwr_liquid_water_path', 
#                     'remote_kazr_cloud_top']
#         avg_list_res = []
#         for avg in avg_list:
#         #     break
#             up = df_nsa_up.loc[:,avg]
#             up_straight = df_nsa_up_straight.loc[:,avg]
#             down = df_nsa_down.loc[:,avg]
#             down_straight = df_nsa_down_straight.loc[:,avg]
#             out = dict(name = avg,
#                        up = up.mean(),
#                        up_std = up.std(),
#                        up_ground2cb = up_straight.mean(),
#                        up_ground2cb_std = up_straight.std(),
#                        down = down.mean(),
#                        down_std = down.std(),
#                        down_ground2cb = down_straight.mean(),
#                        down_ground2cb_std = down_straight.std())
#             avg_list_res.append(out)
#         out_return['avg_list_res'] = copy.deepcopy(avg_list_res)
# ## remove the average values
#         df_nsa_up= df_nsa_up.drop(avg_list, axis=1)
#         df_nsa_down = df_nsa_down.drop(avg_list, axis=1)
    
### get the straight profiles 
    # get the time of the top of the straigh profile -- up
        no_of_ground_cb_connect = 0
        res,top_adjusted, message = get_adjusted_top_up(df_nsa_up, group, launch, top)
        if not res: #(todo) redo for up
            df_nsa_up_straight = df_nsa_up.copy()
            df_nsa_up_straight[:] = np.nan
            send_message(message)
        else:
            no_of_ground_cb_connect+=1
            df_nsa_up_straight = df_nsa.truncate(launch,top_adjusted)
            subgroup = subgroup.append(pd.Series(['top_ground2cloudbase_up'], index=[pd.to_datetime(top_adjusted)], name='type', ))
            
    # get the time of the top of the straigh profile -- down
        res,top_adjusted, message = get_adjusted_top_down(df_nsa_down, group, top, landing)
        if not res: #(todo) redo for up
            df_nsa_down_straight = df_nsa_down.copy()
            df_nsa_down_straight[:] = np.nan
            send_message(message)
        else:
            no_of_ground_cb_connect+=1
            df_nsa_down_straight = df_nsa.truncate(top_adjusted, landing)
            send_message(message)
            subgroup = subgroup.append(pd.Series(['top_ground2cloudbase_down'], index=[pd.to_datetime(top_adjusted)], name='type', ))
            
### get all the relevant avg values
        avg_list = ['ground_atm_pressure', 'remote_ceilometer_cloud_base_altitude', 'cloud_base_transit_altitude',
#                     'remote_mwr_liquid_water_path', 
                    'remote_kazr_cloud_top']
        avg_list_res = []
        for avg in avg_list:
        #     break
            up = df_nsa_up.loc[:,avg]
            up_straight = df_nsa_up_straight.loc[:,avg]
            down = df_nsa_down.loc[:,avg]
            down_straight = df_nsa_down_straight.loc[:,avg]
            out = dict(name = avg,
                       up = up.mean(),
                       up_std = up.std(),
                       up_ground2cb = up_straight.mean(),
                       up_ground2cb_std = up_straight.std(),
                       down = down.mean(),
                       down_std = down.std(),
                       down_ground2cb = down_straight.mean(),
                       down_ground2cb_std = down_straight.std())
            avg_list_res.append(out)
        out_return['avg_list_res'] = copy.deepcopy(avg_list_res)
    ## remove the average values
        df_nsa_up= df_nsa_up.drop(avg_list, axis=1)
        df_nsa_down = df_nsa_down.drop(avg_list, axis=1)
        df_nsa_up_straight= df_nsa_up_straight.drop(avg_list, axis=1)
        df_nsa_down_straight = df_nsa_down_straight.drop(avg_list, axis=1)
        
### turn timeseries into vertical profile

        ts_nsa_up = ts.TimeSeries(df_nsa_up)
        ts_nsa_up_straight = ts.TimeSeries(df_nsa_up_straight)
        ts_nsa_down = ts.TimeSeries(df_nsa_down)
        ts_nsa_down_straight = ts.TimeSeries(df_nsa_down_straight)

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

        vp_nsa_up_straight, vp_nsa_up_straight_std = ts_nsa_up_straight.convert2verticalprofile(altitude_column='altitude', resolution=resolution, return_std=True)
        vp_nsa_down_straight, vp_nsa_down_straight_std = ts_nsa_down_straight.convert2verticalprofile(altitude_column='altitude', resolution=resolution, return_std=True)

        ## the dataset
        ds_out = xr.Dataset()

        ## prepare 1d variables

        vp_nsa_up.data.index.name = 'altitude'
        vp_nsa_up_std.data.index.name = 'altitude'
        vp_nsa_down.data.index.name = 'altitude'
        vp_nsa_down_std.data.index.name = 'altitude'
        
        vp_nsa_up_straight.data.index.name = 'altitude'
        vp_nsa_up_straight_std.data.index.name = 'altitude'
        vp_nsa_down_straight.data.index.name = 'altitude'
        vp_nsa_down_straight_std.data.index.name = 'altitude'

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
            
            df['up_ground2cb'] = vp_nsa_up_straight.data[var]
            df['up_ground2cb_std'] = vp_nsa_up_straight_std.data[var]
            
            df['down'] = vp_nsa_down.data[var]
            df['down_std'] = vp_nsa_down_std.data[var]
            
            df['down_ground2cb'] = vp_nsa_down_straight.data[var]
            df['down_ground2cb_std'] = vp_nsa_down_straight_std.data[var]
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

### add the average remote values
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
### add micellenious
        ds_out['no_of_ground_cb_connect'] = no_of_ground_cb_connect
        subgroup.sort_index(inplace=True)
        subgroup.index.name = 'time'
        ds_out['interesting_times'] = subgroup
        out_return['subgroup'] = subgroup.copy()
        
    # add stuff to follow some unnecessary standards
        ds_out.time.attrs['long_name'] = 'Time offset from base_time'
        ds_out['time_offset'] = ds_out.time.copy()
        
        td = pd.to_datetime(ds_out.time.values[0]) - pd.to_datetime('1970')
        ds_out['base_time'] = int(td.total_seconds())   
        ds_out.base_time.attrs['string'] = pd.to_datetime(ds_out.time.values[0]).__str__() + ' 0:00'
        ds_out.base_time.attrs['long_name'] = 'Base time in Epoch'
        ds_out.base_time.attrs['units'] = 'seconds since 1970-1-1 0:00:00 0:00'
### add productinfo
        
        for key in productinfo:
            ds_out.attrs[key] = productinfo[key]

        # save to netcdf
        if save2netcdf:
            ds_out.to_netcdf(path_out)
        
        if make_quicklooks:
            lower_alt_lim = 20
            aas = []
            f, aa = quicklooks.plot_quicklook(ds_out, plot_cb = True, plot_ct = True, save=save_quicklook, output_path = path_out, lower_alt_lim=lower_alt_lim)
            aas.append(aa)
            f, aa = quicklooks.plot_quicklook(ds_out, plot_cb = True, plot_ct = False, save=save_quicklook, output_path = path_out, lower_alt_lim=lower_alt_lim)
            aas.append(aa)
            f, aa = quicklooks.plot_quicklook(ds_out, plot_cb = False, plot_ct = False, save=save_quicklook, output_path = path_out, lower_alt_lim=lower_alt_lim)
            aas.append(aa)
            
            f, aa = quicklooks.plot_quicklook(ds_out, ground2cloudbase = True, plot_cb = True, plot_ct = True, save=save_quicklook, output_path = path_out, lower_alt_lim=lower_alt_lim)
            aas.append(aa)
            f, aa = quicklooks.plot_quicklook(ds_out, ground2cloudbase = True, plot_cb = True, plot_ct = False, save=save_quicklook, output_path = path_out, lower_alt_lim=lower_alt_lim)
            aas.append(aa)
            f, aa = quicklooks.plot_quicklook(ds_out, ground2cloudbase = True, plot_cb = False, plot_ct = False, save=save_quicklook, output_path = path_out, lower_alt_lim=lower_alt_lim)
            aas.append(aa)
            
            if not show_quicklooks:
                for aa in aas:
                    for aset in aa:
                        for at in aset:
                            at.remove()
                            if hasattr(at,'at'):
                                at.at.remove()
            
        if test:
            break
    
    try:
        out_return['ds_nsascience'] = ds
        out_return['output_path'] = path_out
        out_return['product_info'] = productinfo
        out_return['ds_out'] = ds_out
    except:
        pass
    return out_return