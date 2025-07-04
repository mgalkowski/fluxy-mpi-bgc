import logging
import os
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

def sum_region_fluxes(ds,vars,regions):
    """
    Uses country_fraction and cell_area variables to sum spatial fluxes into
    region/country totals.
    """
    
    #list of variables as input (so not all flux_ and country_flux vars are 
    # scaled if not needed)
    
    return ds

def scale_by_sector_proportions(datadir,ds):
    """
    Produces sector level fluxes by scaling prior and posterior fluxes by the 
    proportional contribution of each sector to the total flux in each grid cell
    (at transport model resolution). Fluxes are then summed over region/country areas
    to produce region/country sector totals.    
    """
    
    sector_prop_path = os.path.join(data_dir,'sector_proportions',
                                       f'{sector_proportions_file}_{species}_yearly_flux_sector_proportions.nc')

    sector_prop = {}

    with xr.open_dataset(sector_prop_path) as f:
        sector_time = f.time.values
        for s in sectors:
            sector_prop[s] = f[f'flux_proportion_{s}']
            
    for s in sectors:
        
        flux_sector_prior_out = np.zeros_like(ds['flux_total_prior'].values)
        flux_sector_post_out = np.zeros_like(ds['flux_total_posterior'].values)
        
        for i,t in enumerate(ds.time.values):
            # matches sector proportions times to inversion times
            if t.astype('datetime64[Y]').astype('datetime64[ns]') in sector_time: 
                t_index = np.where(sector_time == t.astype('datetime64[Y]').astype('datetime64[ns]'))[0][0]
            # if no early enough dates in sector proportion file, use earliest available
            elif t.astype('datetime64[Y]').astype('datetime64[ns]') < sector_time[0]:
                t_index = 0
            # if no late enough dates in sector proportion file, use latest available
            elif t.astype('datetime64[Y]').astype('datetime64[ns]') > sector_time[-1]:
                t_index = -1
            scaling_factor = sector_prop[s].values[t_index,:,:]

            flux_sector_prior_out[i,:,:] = ds[f'flux_total_prior'].values[i,:,:]*scaling_factor
            flux_sector_post_out[i,:,:] = ds[f'flux_total_posterior'].values[i,:,:]*scaling_factor
            
        ds[f'flux_{s}_prior'] = (['time','lat','lon'],sector_flux_out)
        ds[f'flux_{s}_prior'].attrs = {'unit':'mol m-2 s-1',
                                            '_FillValue':np.nan,
                                            'long_name':f'prior {s} {species} flux, created by scaling total flux by sector proportions'}
        
        ds[f'flux_{s}_posterior'] = (['time','lat','lon'],sector_flux_out)
        ds[f'flux_{s}_posterior'].attrs = {'unit':'mol m-2 s-1',
                                            '_FillValue':np.nan,
                                            'long_name':f'posterior total {species} {s} flux, created by scaling total flux by sector proportions'}
        
    #sum over sliced regions to create region/country totals for each sector - using country fraction in flux ds
    # use function above
    
    
    return ds

