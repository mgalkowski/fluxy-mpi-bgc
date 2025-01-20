import numpy as np
import xarray as xr
from matplotlib.dates import MonthLocator, YearLocator
from matplotlib.ticker import NullFormatter
import matplotlib.pyplot as plt
from typing import Literal
from fluxy import config
from fluxy.plots.utils import set_min_decimal_points
import logging

logger = logging.getLogger(__name__)

def plot_mf_timeseries(
        ds_all: dict[str, xr.Dataset],
        specie: str,
        site: str,
        model_colors: dict[str, str],
        config_data: dict[str, dict],
        annotate_coords: dict[int, list],
        ppt_mode: bool = False,
        plot_type: Literal['separate','together'] = 'separate',
        include: dict[str, str | None] = {'Yobs': None,
                                          'Yapost': 'qYapost'},
        diff_include: list[str] = ['Yapriori','Yapost'],
        y_lim: None | list[float] = None
):
    """
    Timeseries plots of observations, modelled mole fractions, baseline mf and/or 
    uncertainties from each model.
    Includes a histogram of the variables plotted in the timeseries plot or 
    the difference between specified variables and observations.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates and for 
            chosen site.
        specie (str): 
            Gas specie, e.g. 'ch4'.
        site (str):
            Obs site, e.g. 'MHD'.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        annotate_coords (dict of lists):
            Coordinates to annotate histogram.
        ppt_mode (logical) (optional):
            If True, adjust annotation position and xlabel rotation to accomodate bigger fonts.
        include (dict of str):
            Dictionary keys are variables to include in the plot.
            The respective values are the uncertainty variables to plot as error bar/uncertainty band.
        diff_include (list of str):
            Variables included in the 'obs - variable' difference histogram, 
            same options as above.
            If empty list, plots the histogram of the variables specified in include.
        y_lim (list of float, optional):
            Mix/max y axis limits to apply to all plots.
    Returns:
        fig (figure): 
            A timeseries and histogram plot for each model included.
    """
      
    models = ds_all.keys()
    specie_info = config_data["species_info"][specie]
    vars_to_plot = include.keys()

    min_mf = np.inf
    max_mf = -np.inf
 
    # Define number of rows in figure
    if plot_type == 'separate':
        nrows = len(models)
    elif plot_type == 'together':
        nrows = 1
    else:
        raise ValueError(f'Option {plot_type} not implemented. Set plot_type to \'separate\' or \'together\'.')
    
    # Create figure
    fig, ax = plt.subplots(nrows, 2, figsize=(15,nrows*3), gridspec_kw={'width_ratios': [0.8,0.2]}, constrained_layout=True)
    
    # Expand axis dimension if 1D
    if nrows == 1: ax = np.expand_dims(ax, axis=0)

    # Loop over all models
    for i,m in enumerate(models):
        
        m0 = m.split('_')[0]
        model_info = config_data["models_info"][m]

        # Define axis index
        if plot_type == 'separate':
            iax = i
        elif plot_type == 'together':
            iax = 0
        
        # Loop over all variables to plot
        for var in vars_to_plot:

            if var not in ds_all[m].keys():
                raise KeyError(f'Variable {var} not found in {m}.')
            
            # Define plotting color
            plot_color = model_colors[m][config.mf_color_index[var]]
            if var == 'Yobs' and len(vars_to_plot) > 1:
                plot_color = 'black'                

            if var == 'Yobs':
                # Make scatter plot
                ax[iax,0].scatter(ds_all[m].time.values,
                                  ds_all[m][var].values,
                                  color=plot_color,
                                  label=f'Obs ({model_info["label"]})',
                                  s=8,
                                  alpha=0.8,
                                  marker='s')        

            else:
                # Make line plot
                ax[iax,0].plot(ds_all[m].time.values,
                               ds_all[m][var].values,
                               color=plot_color,alpha=0.8,
                               linewidth=2.,
                               label=f'{model_info["label"]} {config.mf_labels[var]}')

            unc_var = include[var]

            if unc_var:
                if unc_var not in ds_all[m].keys():
                    raise KeyError(f'Variable {unc_var} not found in {m}.')

                if unc_var[0] == 'q':
                    # Add uncertainty band
                    ax[iax,0].fill_between(ds_all[m].time.values,
                                           ds_all[m][unc_var].values[:,config.model_q_indices[m0][0]],
                                           ds_all[m][unc_var].values[:,config.model_q_indices[m0][1]],
                                           color=plot_color,
                                           alpha=0.2)
                    
                else:
                    # Add error bar
                    ax[iax,0].errorbar(ds_all[m].time.values,
                                       ds_all[m][var].values,
                                       ds_all[m][unc_var].values,
                                       color=plot_color,
                                       alpha=0.4,
                                       fmt='none')                  

        # Get histogram variables and legend
        if len(diff_include) == 0:
            make_diff    = False
            hist_to_plot = vars_to_plot
            legend_hist  = 'Modelled mean'

        else:
            make_diff    = True
            hist_to_plot = diff_include
            legend_hist  = 'Obs - modelled mean'

        # Loop over all variables to plot in histogram
        for v,var in enumerate(hist_to_plot):
            
            if var not in ds_all[m].keys():
                raise KeyError(f'Variable {var} not found in {m}.')
            
            if make_diff:
                var_to_plot = ds_all[m]['Yobs'].values - ds_all[m][var].values
            else:
                var_to_plot = ds_all[m][var].values

            # Plot histogram
            a,b,c = ax[iax,1].hist(var_to_plot,
                                   bins=30,
                                   color=model_colors[m][config.mf_color_index[var]],
                                   density=1
                                   )
            if make_diff:
                ax[iax,1].vlines(0,0,np.max(a),color='dimgrey',linewidth=3.)
            
            if plot_type == 'separate':
                index = v
            elif plot_type == 'together':
                index = i
            
            # Compute and format mean and std of the histogram
            var_mean = np.nanmean(var_to_plot)
            var_std = np.nanstd(var_to_plot)
            str_mean = set_min_decimal_points(var_mean)
            str_std = set_min_decimal_points(var_std)

            # Write mean/std to histogram
            ax[iax,1].annotate(f'$\mu$: {str_mean}\n$\sigma$: {str_std}',
                               xy=annotate_coords[index],
                               xycoords='axes fraction',
                               color=model_colors[m][config.mf_color_index[var]]
                               )

        # Write number of obs
        if plot_type == 'separate':
            n_obs = (~np.isnan(ds_all[m]['Yobs'].values)).sum()
            if (ppt_mode):
                pos_xy = [0.57,1.05]
            else:
                pos_xy = [0.65,1.05]
            ax[iax,1].annotate('$N_{obs}$: '+str(n_obs),
                               xy=pos_xy,
                               xycoords='axes fraction',
                               color='k'
                               )

        # Set histogram x-axis label
        ax[iax,1].set_xlabel(legend_hist)
    
        # Get timeseries y-axis minimum and maximum
        min_mf = min(min_mf, ax[iax,0].get_ylim()[0])
        max_mf = max(max_mf, ax[iax,0].get_ylim()[1])
        
        # Set timeseries title
        if plot_type == 'separate':
            plot_title = model_info["label"]
        elif plot_type == 'together':
            plot_title = 'All models'
        ax[iax,0].set_title(plot_title)

        # Set timeseries y-axis label and legend
        ax[iax,0].set_ylabel(f'{specie_info["species_print"]} {site} ({specie_info["mf_units_print"]})')
        leg = ax[iax,0].legend(ncol=2,borderpad=.2,columnspacing=1.0)
        try:
            for l in leg.legend_handles:
                l.set_linewidth(5.0)
        except:
            for l in leg.legendHandles:
                l.set_linewidth(5.0)
        
        # Set timeseries x-axis ticks
        if int(ds_all[m].time.values[-1].astype('datetime64[M]')-ds_all[m].time.values[0].astype('datetime64[M]')) > 12:
            ax[iax,0].xaxis.set_minor_locator(MonthLocator())
            ax[iax,0].xaxis.set_minor_formatter(NullFormatter())
            ax[iax,0].xaxis.set_major_locator(YearLocator())
        else:
            ax[iax,0].xaxis.set_major_locator(MonthLocator())
            if (ppt_mode):
                ax[iax,0].tick_params(axis='x', rotation=70)

    # Set timeseries y-axis min/max                
    if y_lim == None:
        for ax0 in ax[:,0]:
            ax0.set_ylim([min_mf - 0.02*min_mf, max_mf + 0.05*max_mf])
    else:
        for ax0 in ax[:,0]:
                ax0.set_ylim(y_lim)
            
    logger.info('If annotations in the histograms are not displaying correctly, adjust annotate_coords.')
    
    return fig

def plot_obs_diff(ds_all,species,site,
                               model_colors,s_data,m_data,annotate_coords,ppt_mode=False,
                               include=['Yapost'],
                               diff_include=['Yapost'],
                               y_lim=None):
    """
    Plot of the absolute difference between variables from two models.
    Also includes a histogram for each model, showing the difference between
    the 'include' variables fit to the observations.
    
    If more than two models are included in ds_all, only the first two
    models will be plotted.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates and for 
            chosen site.
        species (str): 
            Gas species, e.g. 'ch4'.
        site (str):
            Obs site, e.g. 'MHD'.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        annotate_coords (dict of lists):
            Coordinates to annotate histogram.
        ppt_mode (logical) (optional):
            If True, adjust xlabel rotation to accomodate bigger fonts.
        include (list of str):
            Variables included in the plot, options for 'Yobs', 'Yapriori',
            'Yapost', 'YaprioriBC', 'YapostBC'.
        diff_include (list of str):
            Variables included in the 'obs - variable' difference histogram, 
            same options as above.
        y_lim (list of float, optional):
            Mix/max y axis limits to apply to all plots.
    Returns:
        fig (figure): 
            A timeseries and histogram plot for each model included.
    """

    var_labels = {'Yapriori':'prior mf',
                  'Yapost':'posterior mean mf',
                  'YaprioriBC':'prior baseline',
                  'YapostBC':'posterior mean baseline',
                  'Yapriori_bias':'prior bias',
                  'Yapost_bias':'posterior bias',
                  'YaprioriOUTER':'prior outer region mf',
                  'YapostOUTER':'posterior outer region mf',
                  'Yobs':'observed mf',
                  'uYobs_repeatability':'obs repeatability mf uncertainty',
                  'uYobs_variability':'obs variability mf uncertainty',
                  'uYmod':'model uncertainty',
                  'uYtotal':'total uncertainty'}
    var_colors = {'Yapriori':1,
                  'Yapost':0,
                  'YaprioriBC':1,
                  'YapostBC':0,
                  'Yapriori_bias':0,
                  'Yapost_bias':1,
                  'YaprioriOUTER':1,
                  'YapostOUTER':0,
                  'Yobs':0,
                  'uYobs_repeatability':0,
                  'uYobs_variability':0,
                  'uYmod':1,
                  'uYtotal':1}
        
    models = list(ds_all.keys())
    min_mf = []
    max_mf = []
        
    fig = plt.figure(constrained_layout=True,figsize=(15,7))
    gs = fig.add_gridspec(len(models),2,width_ratios=[0.8,0.2])

    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    both_times0 = np.isin(ds_all[models[0]].time.values,ds_all[models[1]].time.values)
    both_times1 = np.isin(ds_all[models[1]].time.values,ds_all[models[0]].time.values)
    
    ds_all[models[0]] = ds_all[models[0]].sel(time=both_times0)
    ds_all[models[1]] = ds_all[models[1]].sel(time=both_times1)
    
            
    for var in include:
        try:
            ax.scatter(ds_all[models[0]].time.values,
                       ds_all[models[0]][var].values - ds_all[models[1]][var].values,
                       color=model_colors[models[0]][var_colors[var]],alpha=0.5,
                       label=f'{m_data[models[0]]["label"]} - {m_data[models[1]]["label"]}\n{var_labels[var]}',
                       linewidth=2,s=8)

        except:
            # handle old ncdf files
            if var == 'uYmod':
                m00 = models[0].split('_')[0]
                m01 = models[1].split('_')[0]

                try:
                    uYmod0 = ds_all[models[0]]['Yobs'].values - ds_all[models[0]]['qYmod'].values[:,config.model_q_indices[m00][0]]
                    uYmod1 = ds_all[models[1]]['Yobs'].values - ds_all[models[1]]['qYmod'].values[:,config.model_q_indices[m01][0]]

                    ax.scatter(ds_all[models[0]].time.values,
                                uYmod0 - uYmod1,
                                color=model_colors[models[0]][var_colors[var]],alpha=0.5,
                                label=f'{m_data[models[0]]["label"]} - {m_data[models[1]]["label"]}\n{var_labels[var]}',
                                linewidth=2,s=8)
                    print(f'WARNING: uYmod is not present in both models. This quantity is being computed from qYmod.')

                except:
                    print(f'ERROR: {models[0]} and {models[1]} have different definitions of uYmod!')

            elif var == 'uYobs_repeatability':
                try:
                    ax.scatter(ds_all[models[0]].time.values,
                                ds_all[models[0]]['uYobs'].values - ds_all[models[1]]['uYobs'].values,
                                color=model_colors[models[0]][var_colors[var]],alpha=0.5,
                                label=f'{m_data[models[0]]["label"]} - {m_data[models[1]]["label"]}\n{var_labels[var]}',
                                linewidth=2,s=8)
                    print(f'WARNING: uYobs_repeatability is not present in both models. uYobs is being plotted instead.')

                except:
                    print(f'ERROR: {models[0]} and {models[1]} have different definitions of uYobs!')

            else:
                print(f'ERROR: variable {var} not found or deprecated in {models[0]} or {models[1]}!')

        #if var == 'Yapost':
        #    ax.fill_between(ds_all[m].time.values,
        #                    ds_all[m]['qYapost'].values[:,config.model_q_indices[m0][0]],
        #                    ds_all[m]['qYapost'].values[:,config.model_q_indices[m0][1]],
        #                    color=model_colors[m][var_colors[var]],alpha=0.3)
                #if var == 'YapostBC':
                #    ax.fill_between(ds_all[m].time.values,
                #                    ds_all[m]['qYapostBC'].values[:,config.model_q_indices[m][0]],
                #                    ds_all[m]['qYapostBC'].values[:,config.model_q_indices[m][1]],
                #                    color=model_colors[m][var_colors[var]],alpha=0.5)
        
    for i,m in enumerate(models):
        
        m0 = m.split('_')[0]

        # Plot histogram
        if len(diff_include) == 0:
            make_diff   = False
            vars        = include
            legend_hist = 'Modelled mean'

        else:
            make_diff   = True
            vars        = diff_include
            legend_hist = 'Obs - modelled mean'

        for v,var in enumerate(vars):

            if make_diff:
                var_plot = ds_all[m]['Yobs'].values - ds_all[m][var].values
            else:
                try:
                    var_plot = ds_all[m][var].values
                except:
                    if var == 'uYmod':
                        var_plot = ds_all[m]['Yobs'].values - ds_all[m]['qYmod'].values[:,config.model_q_indices[m0][0]]
                    elif var == 'uYobs_repeatability':
                        var_plot = ds_all[m]['uYobs'].values
                    else:
                        continue
            
            if np.nanmean(var_plot) <= 0.01:
                var_mean = np.round(np.nanmean(var_plot),5)
                var_sd = np.round(np.nanstd(var_plot),5)
            else:
                var_mean = np.round(np.nanmean(var_plot),2)
                var_sd = np.round(np.nanstd(var_plot),2)
            
            a,b,c = ax2.hist(var_plot,bins=30,color=model_colors[m][var_colors[var]],density=1,alpha=0.7)
            if make_diff:
                ax2.vlines(0,0,np.max(a),color='dimgrey',linewidth=3.)
            
            with np.printoptions(precision=2, suppress=True):

                ax2.annotate('$\mu$: '+str(var_mean)+'\n$\sigma$: '+str(var_sd),xy=annotate_coords[i],
                                xycoords='axes fraction',color=model_colors[m][var_colors[var]])
        
    ax2.set_xlabel(legend_hist)

    min_mf.append(ax.get_ylim()[0])
    max_mf.append(ax.get_ylim()[1])
    
    ax.set_title(f'{m_data[models[0]]["label"]} - {m_data[models[1]]["label"]}')
    ax.set_ylabel(f'{s_data[species]["species_print"]} {site} ({s_data[species]["mf_units_print"]})')
    leg = ax.legend(ncol=2,borderpad=.2,columnspacing=1.0)
    try:
        for l in leg.legend_handles:
            l.set_linewidth(5.0)
    except:
        for l in leg.legendHandles:
            l.set_linewidth(5.0)
    
    if int(ds_all[m].time.values[-1].astype('datetime64[M]')-ds_all[m].time.values[0].astype('datetime64[M]')) > 12:
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(YearLocator())
    else:
        ax.xaxis.set_major_locator(MonthLocator())
        if (ppt_mode):
            ax.tick_params(axis='x', rotation=70)
        
    if y_lim is None:
        ax.set_ylim([min(min_mf)-(0.02*min(min_mf)),
                                max(max_mf)+(0.05*max(max_mf))])
    else:
        ax.set_ylim(y_lim)
        
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim')
    print('NOTE: If annotations in the histograms are not displaying correctly, adjust annotate_coords.')
    
    return fig


def plot_sites_timeseries(ds_all,var,start_date,end_date,model_colors,m_data):
    """
    Plot the timeseries of data available for each site and model.
    
    Args:
        ds_all : 
            Dictionnary of xarray returned by read_mf
        var : 
            Var for which the timeseries should be plotted
        start_date (str): 
            Date to plot data from, e.g. '2021-01-01'
        end_date (str): 
            Date to plot data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
    """
    siteList = np.sort(np.unique(np.concatenate([ds_all[m].sitenames.values.astype(str) for m in ds_all.keys()])))

    fig,ax = plt.subplots(1,1,figsize = (0.7*len(siteList),8))
    leg = []
    for iSite,site in enumerate(siteList):
        if iSite!=0:
            ax.plot([iSite-0.5,iSite-0.5],[np.datetime64(start_date),np.datetime64(end_date)],
                   c='gray',ls='-',lw=1)
        for i,m in enumerate(ds_all.keys()):
            try:
                if m not in leg:
                    site_index = np.where(ds_all[m]['sitenames'].astype(str) == site)[0][0]
                    data = ds_all[m].isel(nsite=site_index)[var].dropna(dim='time').time.values
                    ax.scatter(-2*np.ones(data.size),
                               data,c=model_colors[m][0],s=20,label=m_data[m]["label"])
                    leg.append(m)

                site_index = np.where(ds_all[m]['sitenames'].astype(str) == site)[0][0]
                data = ds_all[m].isel(nsite=site_index)[var].dropna(dim='time').time.values
                ax.scatter((iSite+0.2*(i-1))*np.ones(data.size),
                           data,c=model_colors[m][0],s=2)

            except:
                pass
    ax.set_ylim(np.datetime64(start_date)-np.timedelta64(1,'D'),
                np.datetime64(end_date)+np.timedelta64(1,'D'))
    
    
    ax.set_xticks(np.arange(siteList.size))
    ax.set_xticklabels(siteList)
    
    
    if int(np.datetime64(end_date).astype('datetime64[M]')-np.datetime64(start_date).astype('datetime64[M]')) > 12:
        ax.yaxis.set_minor_locator(MonthLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(YearLocator())
    else:
        ax.yaxis.set_major_locator(MonthLocator())
    ax.yaxis.grid(True, which='major')
    
    ax.set_xlim(-1,siteList.size)
        
    plt.legend(loc='upper right')
    
    return fig
