import matplotlib.pyplot as plt
import numpy as np

def plot_avg(a, variable, direction = 'up', show_std = True, **kwargs):
    df = variable.to_pandas()
    if direction == 'up':
        df_mean = df.up
        df_std = df.up_std
    elif direction == 'down':
        df_mean = df.down
        df_std = df.down_std
    
    
    a.axhline(df.up, **kwargs)
    if show_std:
        a.axhspan(df_mean - df_std, df_mean + df_std, alpha = 0.5, **kwargs)

def plot_it(a, ds, variable, direction='up' , color = None, next_color=None, cloud_base = True, cloud_top = True, **kwargs):
    
    df = variable.to_pandas()
    if direction == 'up':
        df_mean = df.up
        df_std = df.up_std
    elif direction == 'down':
        df_mean = df.down
        df_std = df.down_std
        
    if not isinstance(next_color, type(None)):
        for i in range(next_color):
            a._get_lines.get_next_color()
        
    a.plot(df_mean, df.index, color = color)
    g = a.get_lines()[-1]
    color = g.get_color()
    g.set_solid_joinstyle('round')#'miter', 'round', 'bevel')
    g.set_solid_capstyle('round')
    pc = a.fill_betweenx(df.index, df_mean-df_std, df_mean + df_std, alpha = 0.5, facecolor = color)
    
    if cloud_base:
        plot_avg(a, ds.remote_ceilometer_cloud_base_altitude, direction = direction, color = '0.6')
    if cloud_top:
        plot_avg(a, ds.remote_kazr_cloud_top, direction = direction, show_std = False, color = '0.6', ls = '--')
    out = {}
    out['g'] = g
    out['pc'] = pc
    return out
    
    

def plot_set(ds, direction = 'up', plot_cb = True, plot_ct = True,  ax = None, alt = False ):
    """plot a row of axes with the relevant parameters"""
    if isinstance(ax, type(None)):
        f, aa = plt.subplots(1,5,gridspec_kw={'wspace':0.02},
                             sharey=True, 
                            
                            )
        f.set_figwidth(f.get_figwidth() * 1.5)
    else:
        aa = ax
    a_tmp = aa[0]
    a_rh = aa[1]
    a_nc = aa[3]
    a_md = aa[4] # mean diameter
    a_lwp = aa[2] # liquid water path

    # temperature and pressure
    out = plot_it(a_tmp, ds, ds.temperature, direction=direction, cloud_base = plot_cb, cloud_top = plot_ct)
    at = a_tmp.twiny()
    a_tmp.at = at
    
    g = out['g']
    g.set_label('Temp.')
    out = plot_it(at, ds, ds.temperature_potential, direction=direction, next_color=1, cloud_base = False, cloud_top = False)
    gtp = out['g']
    gtp.set_label('Pot. temp.')
    out = plot_it(at, ds, ds.temperature_equiv_potential, direction=direction, next_color=1, cloud_base = False, cloud_top = False)
    gtep = out['g']
    gtep.set_label('Eq. pot. temp')

    a_tmp.set_xlabel('T (C°)')
    at.set_xlabel('$\\theta$')
    a_tmp.legend(handles = [g, gtp, gtep], loc = 9, fontsize = 'x-small')
    # rh
    plot_it(a_rh, ds, ds.relative_humidity, direction=direction, cloud_base = plot_cb, cloud_top = plot_ct)
    a_rh.set_xlabel('RH (%)')

    #lwp & precip
    out = plot_it(a_lwp, ds, ds.remote_mwr_liquid_water_path, direction=direction, cloud_base= plot_cb, cloud_top = plot_ct)
    glwp = out['g']
    glwp.set_label('lwp')
    at = a_lwp.twiny()
    a_lwp.at = at
    
    out = plot_it(at, ds, ds.ground_precip_rate, direction=direction, next_color=1, cloud_base = False, cloud_top = False)
    gpc = out['g']
    gpc.set_label('precip.')

    a_lwp.set_xlabel('liquid water path')
    at.set_xlabel('Pricip. rate (mm/hr)')

    a_lwp.legend(handles = [glwp, gpc], loc = 9)

    # nc
    out = plot_it(a_nc, ds, ds.pops_particle_number_concentration, direction=direction, cloud_base = plot_cb, cloud_top = plot_ct)
    gp = out['g']
    gp.set_label('POPS')
    out = plot_it(a_nc, ds, ds.ground_uhsas_particle_number_concentration_pops_overlap, direction=direction, cloud_base = False, cloud_top = False)
    gu = out['g']
    gu.set_label('UHSAS')

    a_nc.set_xlabel('Particle number\nconc. (dN/dlog(Dp))')
    a_nc.legend(handles = [gp,gu], fontsize = 'small', loc = 9)
    # mean diameter
    plot_it(a_md, ds, ds.pops_particle_mean_diameter, direction=direction, cloud_base = plot_cb, cloud_top = plot_ct)

    a_md.set_xlabel('Mean diameter\n(nm)')

    # global plot settings
    aa[0].set_ylabel('Altitude (m)')
    for a in aa:
        a.grid(ls = '--')
    for a in aa[1:]:
        a.tick_params(left = False)
    return aa

def get_plot_lim(ax,lower_alt_lim = 20):
    mins = []
    maxs = []
    for g in ax.get_lines():
        x = g.get_xdata()
        y = g.get_ydata()
        if isinstance(y,list):
            assert(len(y) == 2)
            continue
        ll_arg = (y > lower_alt_lim).argmax()
        
        # remove nans
        xnn = x[ll_arg:]
        xnn = xnn[~np.isnan(xnn)]
        if xnn.shape[0] == 0:
            continue
        mins.append(xnn.min())
        maxs.append(xnn.max())
    #     break
    #     print('done')
    if len(mins) == 0:
        lims = (None, None)
    else:
        lims = (np.min(mins), np.max(maxs))
    return lims

def plot_quicklook(ds, plot_cb = True, plot_ct = True, save = True, output_path = None, lower_alt_lim = None, overwrite = False):
    """
    

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    plot_cb : TYPE, optional
        DESCRIPTION. The default is True.
    plot_ct : TYPE, optional
        DESCRIPTION. The default is True.
    save : TYPE, optional
        DESCRIPTION. The default is True.
    output_path : TYPE, optional
        This is actually the outputpath of the nsascience_vp file not the 
        qicklook
    lower_alt_lim: float, optional
        ignore values below this altitude when scaling the plots. This is to 
        remove effects from contaminated data close to the ground

    Returns
    -------
    f : TYPE
        DESCRIPTION.
    aa : TYPE
        DESCRIPTION.

    """
    # ds = out['ds_out']
    f,aa = plt.subplots(2,5, gridspec_kw={'wspace':0.02, 'hspace': 0.4}, sharey=True, )
    f.set_figheight(f.get_figheight() * 2)
    f.set_figwidth(f.get_figwidth() * 1.5)
    
    plot_set(ds, ax = aa[0], plot_cb = plot_cb, plot_ct = plot_ct)
    plot_set(ds,direction='down', ax = aa[1], plot_cb = plot_cb, plot_ct = plot_ct)
    
    if not isinstance(lower_alt_lim, type(None)):
        for aset in aa:
            for at in aset:
                lim = get_plot_lim(at, lower_alt_lim = lower_alt_lim)
                at.set_xlim(lim)
                if hasattr(at,'at'):
                    lim = get_plot_lim(at.at, lower_alt_lim = lower_alt_lim)
                    at.at.set_xlim(lim)
    
    
    if save:
        # output_path = out['output_path']
        figname = output_path.name.replace('.nc', '.quicklook.png')
    
        without = []
        if plot_cb and plot_ct:
            what = 'all'
        else:
            if not plot_cb:
                without.append('cb')
            if not plot_ct:
                without.append('ct')
            what = 'wo_' + '_'.join(without)
        
        if not isinstance(lower_alt_lim, type(None)):
            what += f'_ll{lower_alt_lim}m'
        
        path2quicklooks = output_path.parent.joinpath('quicklooks')
        path2quicklooks.mkdir(exist_ok=True)
        path2quicklooks = path2quicklooks.joinpath(what)
        path2quicklooks.mkdir(exist_ok=True)
    
        path2figure = path2quicklooks.joinpath(figname)
        if path2figure.is_file() and (overwrite == False):
            print('file exists ... skip')
        else:
            f.patch.set_alpha(0)
            f.savefig(path2figure, bbox_inches = 'tight')
            
    return f,aa