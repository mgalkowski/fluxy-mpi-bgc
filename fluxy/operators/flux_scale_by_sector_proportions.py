import logging
import os
import numpy as np
import xarray as xr
from pathlib import Path
from fluxy.operators.convert import get_units_conversion_factor

logger = logging.getLogger(__name__)

def scale_by_sector_proportions(data_dir:str,
                                ds: xr.Dataset,
                                species: str,
                                sectors: list[str],
                                regions: list[str],
                                country_flux_units_print: str,
                                config_data = dict[str, dict],
                                sector_file: str = 'EUROPE_EDGAR',
                                create_region_sector_totals: bool = True
) -> xr.Dataset:
    """
    Produces sector level fluxes by scaling prior and posterior fluxes by the 
    proportional contribution of each sector to the total flux in each grid cell
    (at transport model resolution). Fluxes are then summed over region/country areas
    to produce region/country sector totals.    
    
    Args:
        data_dir (str):
            Path to top data directory.
        ds (xarray dataset):
            xarray dataset with model data.
        species (str):
            Gas species, e.g. 'ch4'.
        sectors (list of str):
            Emissions sectors to include, options for 'agriculture', 'waste', 
            'energy' and 'industry'.
        regions (list of str):
            Region names. Sector region/country totals are only calculated for
            these areas.
        country_flux_units_print (str):
            Units for country flux, e.g. 'Gg yr-1'
        config_data (dict):
            config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        sector_file (str):
            Start of sector file name, e.g. 'EUROPE_EDGAR'
        create_region_sector_totals (bool):
            If True, sums spatial fluxes over country_fraction masks 
            to create country/region totals.
    
    Returns:
        ds (xarray dataset):
            Input ds, with added flux_sector_prior and flux_sector_posterior variables.
            If regions is not None, also contains region/country sector total variables.
    """
    
    if create_region_sector_totals == True:
        
        if 'country_fraction' not in ds:
            raise ValueError(
                f"country_fraction variable not present in dataset for {ds['inversion_system']} "
                "Cannot add sector country totals."
                )
            
        if 'cell_area' in ds:
            cell_area = ds['cell_area'].values
        else:
            parent_dir = Path(__file__).parent.parent.parent
            configs_dir = parent_dir / "configs"
            if 'domain' in ds.attrs:
                domain = ds.attrs["domain"]
            else:
                domain = sector_file.split('_')[0]
                logger.warning(f'No domain info in dataset attributes, so reading domain from sector_filename: {domain}')
            with xr.open_dataset(os.path.join(configs_dir,f"{domain}_cell_area.nc")) as f:
                cell_area = f['cell_area'].values
                
        dict_regions: dict[str,str] = config_data['regions_info'].get("regions",{})
                
        country_codes = config_data['regions_info'].get("country_codes", {})
        country_search = [country_codes.get(r, r) for r in regions]
                
        #units for spatial flux to regional flux totals
        molar_mass = config_data["species_info"][species]["molar_mass"]
        s_in_year = 60*60*24*365
        scaling_factor = get_units_conversion_factor('mol s-1', country_flux_units_print, molar_mass)
     
    sector_prop_path = os.path.join(data_dir,'sector_flux',
                                       f'{sector_file}_{species}_yearly_flux_sectors.nc')

    sector_prop = {}
    
    with xr.open_dataset(sector_prop_path) as f:
        sector_time = f.time.values
        for s in sectors:
            #sector_prop[s] = f[f'flux_proportion_{s}']
            sector_prop[s] = f[f'flux_{s}']/f[f'flux_total']
            
    if create_region_sector_totals == True:
        
        # create a dictionary of {region_name:[available_region_masks]}  
        country_all = {}

        for country in country_search:
            # if a grouped region flux total and grouped country mask is available e.g. InTEM: {BEL-LUX-NLD:[BEL-LUX-NLD]}
            if country in ds["country_fraction"].country and country in ds['country'].values:
                country_all[country] = [country]
                country_id = np.where(ds["country_fraction"].country.values == country)[0]
                r_fraction = ds['country_fraction'].values[country_id,:,:]
                
                # if a grouped region flux total but no grouped country mask is available, e.g. RHIME: {BEL-LUX-NLD:[BEL,LUX,NLD]}
                if np.isnan(r_fraction).all() == True:
                    logger.warning(
                        f"{country} country_fraction is not present in ds. "+
                        f"Considering sum of individual countries: {country_all}."
                        )
                    country_all[country] = country.split('-')
            else:
                # if no region flux total and no country mask is available, e.g. ELRIS: {BEL:BEL,LUX:LUX,NLD:NLD}
                logger.warning(
                    f"{country} country_fraction is not present in ds. "+
                    f"Considering sum of individual countries: {country_all}."
                    )
                for c_add in country.split('-'):
                    country_all[c_add] = [c_add]
            
    for s_id,s in enumerate(sectors):
        
        flux_sector_prior_out = np.zeros_like(ds['flux_total_prior'].values)
        flux_sector_post_out = np.zeros_like(ds['flux_total_posterior'].values)
        
        region_sector_prior_out = np.zeros_like(ds['flux_total_prior_country'].values)
        region_sector_post_out = np.zeros_like(ds['flux_total_posterior_country'].values)
        
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

            if create_region_sector_totals == True:
                #TODO: add calculation of region/country sector flux uncertainties here
                #      currently just calculates flux prior and posterior mean
                
                # loop through only selected regions and their matching region masks
                for grouped_c in country_all.keys():
                    c_update = np.where(ds['country'].values == grouped_c)[0]
                    
                    for c in country_all[grouped_c]:                        
                        country_id = np.where(ds["country_fraction"].country.values == c)[0]
                        r_fraction = ds['country_fraction'].values[country_id,:,:]
                        cell_area_by_country = cell_area * r_fraction
                        
                        region_sector_prior_out[i,c_update] += np.nansum(flux_sector_prior_out[i,:,:]*r_fraction
                                                            *cell_area_by_country*scaling_factor)
                        region_sector_post_out[i,c_update] += np.nansum(flux_sector_post_out[i,:,:]*r_fraction
                                                                *cell_area_by_country*scaling_factor)
                    
        ds[f'flux_{s}_prior'] = (['time','latitude','longitude'],flux_sector_prior_out)
        ds[f'flux_{s}_prior'].attrs = {'units':'mol m-2 s-1',
                                            '_FillValue':np.nan,
                                            'long_name':f'prior {s} {species} flux, created by scaling total flux by sector proportions'}
        
        ds[f'flux_{s}_posterior'] = (['time','latitude','longitude'],flux_sector_post_out)
        ds[f'flux_{s}_posterior'].attrs = {'units':'mol m-2 s-1',
                                            '_FillValue':np.nan,
                                            'long_name':f'posterior total {species} {s} flux, created by scaling total flux by sector proportions'}
        
        if create_region_sector_totals == True:
        
            ds[f'flux_{s}_prior_country'] = (['time','country'],region_sector_prior_out)
            ds[f'flux_{s}_prior_country'].attrs = {'units':'kg yr-1',
                                                '_FillValue':np.nan,
                                                'long_name':f'country {s} {species} prior flux, created by scaling total flux by sector proportions'}

            ds[f'flux_{s}_posterior_country'] = (['time','country'],region_sector_post_out)
            ds[f'flux_{s}_posterior_country'].attrs = {'units':'kg yr-1',
                                                '_FillValue':np.nan,
                                                'long_name':f'country {s} {species} posterior flux, created by scaling total flux by sector proportions'}

    return ds

