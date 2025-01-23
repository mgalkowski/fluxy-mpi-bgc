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
        plot_type: Literal['separate','together','diff'] = 'separate',
        include: dict[str, str | None] = {'Yobs': None,
                                          'Yapost': 'qYapost'},
        diff_include: list[str] | None = None,
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
            Variables included in the 'obs - variable' difference histogram.
            If None, plots the histogram of the variables specified in include.
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
    elif plot_type in ['together', 'diff']:
        nrows = 1
    else:
        raise ValueError(f'Option {plot_type} not implemented. Set plot_type to \'separate\', \'together\' or \'diff\'.')
    
    # Create figure
    fig, ax = plt.subplots(nrows, 2, figsize=(15,nrows*3), gridspec_kw={'width_ratios': [0.8,0.2]}, constrained_layout=True)
    
    # Expand axis dimension if 1D
    if nrows == 1: ax = np.expand_dims(ax, axis=0)

    # Loop over all models
    for i,m in enumerate(models):
        
        m0 = m.split('_')[0]

        # Define plot_type specific settings
        if plot_type == 'separate':
            iax = i #axis index
            model_label = config_data["models_info"][m]["label"]
            model_color = model_colors[m]
        elif plot_type == 'together':
            iax = 0
            model_label = config_data["models_info"][m]["label"]
            model_color = model_colors[m]
        elif plot_type == 'diff':
            iax = 0
            mdiff0, mdiff1 = m.split('-')
            model_label = f'{config_data["models_info"][mdiff0]["label"]} - {config_data["models_info"][mdiff1]["label"]}'
            model_color = model_colors[mdiff0]
        
        # Loop over all variables to plot
        for var in vars_to_plot:

            if var not in ds_all[m].keys():
                raise KeyError(f'Variable {var} not found in {m}.')
            
            # Define plotting color
            plot_color = model_color[config.mf_color_index[var]]
            if var == 'Yobs' and len(vars_to_plot) > 1:
                plot_color = 'black'                

            if var == 'Yobs' or plot_type == 'diff':
                # Make scatter plot
                ax[iax,0].scatter(ds_all[m].time.values,
                                  ds_all[m][var].values,
                                  color=plot_color,
                                  label=f'{model_label} {config.mf_labels[var]}',
                                  s=8,
                                  alpha=0.8,
                                  marker='s')        

            else:
                # Make line plot
                ax[iax,0].plot(ds_all[m].time.values,
                               ds_all[m][var].values,
                               color=plot_color,alpha=0.8,
                               linewidth=2.,
                               label=f'{model_label} {config.mf_labels[var]}')

            unc_var = include[var]

            if unc_var:
                if plot_type == 'diff':
                    raise ValueError(f'Option plot_type=\'diff\' does not accept uncertainties. Replace \'{unc_var}\' by None.')

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
        
        # Plot histogram
        plot_histogram(ax[iax,1],
                       ds_all[m],
                       m,
                       vars_to_plot,
                       diff_include,
                       model_color,
                       ppt_mode,
                       annotate_coords,
                       annotate_index=i,
                       plot_type=plot_type)

        # Get timeseries y-axis minimum and maximum
        min_mf = min(min_mf, ax[iax,0].get_ylim()[0])
        max_mf = max(max_mf, ax[iax,0].get_ylim()[1])
        
        # Set timeseries title
        if plot_type in ['separate','diff']:
            plot_title = model_label
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
    if y_lim is None:
        for ax0 in ax[:,0]:
            ax0.set_ylim([min_mf - 0.02*min_mf, max_mf + 0.05*max_mf])
    else:
        for ax0 in ax[:,0]:
                ax0.set_ylim(y_lim)
            
    logger.info('If annotations in the histograms are not displaying correctly, adjust annotate_coords.')
    
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

def plot_histogram(axis,
                   ds: xr.Dataset,
                   model: str,
                   vars_to_plot: list[str],
                   diff_include: list[str] | None,
                   model_color: list[str],
                   ppt_mode: bool,
                   annotate_coords: dict[int, list],
                   annotate_index: int,
                   plot_type: Literal['separate','together']
):

    # Get histogram variables and legend
    if diff_include:
        hist_to_plot = diff_include
        legend_hist  = 'Obs - Plotted variable'
    else:
        hist_to_plot = vars_to_plot
        legend_hist  = 'Plotted variable'

    # Loop over all variables to plot in histogram
    for v,var in enumerate(hist_to_plot):
        
        if var not in ds.keys():
            raise KeyError(f'Variable {var} not found in {model}.')
        
        if diff_include:
            var_to_plot = ds['Yobs'].values - ds[var].values
        else:
            var_to_plot = ds[var].values

        # Plot histogram
        a,b,c = axis.hist(var_to_plot,
                          bins=30,
                          color=model_color[config.mf_color_index[var]],
                          density=1
                          )

        if diff_include:
            axis.vlines(0,0,np.max(a),color='dimgrey',linewidth=3.)
        
        if plot_type == 'separate':
            index = v
        elif plot_type in ['together', 'diff']:
            index = annotate_index
        
        # Compute and format mean and std of the histogram
        var_mean = np.nanmean(var_to_plot)
        var_std = np.nanstd(var_to_plot)
        str_mean = set_min_decimal_points(var_mean)
        str_std = set_min_decimal_points(var_std)

        # Write mean/std to histogram
        axis.annotate(f'$\mu$: {str_mean}\n$\sigma$: {str_std}',
                      xy=annotate_coords[index],
                      xycoords='axes fraction',
                      color=model_color[config.mf_color_index[var]]
                      )

    # Write number of obs
    if plot_type == 'separate':
        n_obs = ds['Yobs'].count().values
        if (ppt_mode):
            pos_xy = [0.57,1.05]
        else:
            pos_xy = [0.65,1.05]

        axis.annotate('$N_{obs}$: '+str(n_obs),
                      xy=pos_xy,
                      xycoords='axes fraction',
                      color='k'
                      )

    # Set histogram x-axis label
    axis.set_xlabel(legend_hist)

    return None
