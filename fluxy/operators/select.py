import xarray as xr
import numpy as np
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def convert_molar_to_mass_flux(ds, M):
    """
    Converts spatial flux variables in the flux dataset from mol/m²/s to kg/km²/year.
    
    Args:
        ds (xarray dataset): The xarray dataset containing flux variables.
        M (float): The molar mass in g/mol.
    """
    
    # Convert the molar mass to kg/mol
    M_kg = M * 0.001  # Convert grams to kilograms
    
    # List to store names of converted variables
    converted_vars = []
    
    for var_name, variable in ds.items():
        if 'units' in variable.attrs and variable.attrs['units'] == 'mol m-2 s-1':
            # Convert the variable data
            # Multiply by molar mass, seconds per year, and convert m² to km²
            ds[var_name] = variable * M_kg * 31.536e6 # kg/km²/year
            ds[var_name].attrs['units'] = 'kg km-2 yr-1' # Update the units
            
            # Add the variable name to the list of converted variables
            converted_vars.append(var_name)

    print("Converting molar flux variables to mass flux (kg km-2 yr-1)")
    
    return ds

def scale_mf(model, ds_model, specie_info):
        
    logger.info(f'Dividing {model} mole fractions by {specie_info["mf_units_scaling"]}')
    var_names = []

    # Find mole fraction variables in dataset
    for k in ds_model.keys():
        if 'units' in ds_model[k].attrs.keys():
            if ds_model[k].attrs['units'] == 'mol mol-1':
                var_names.append(k)

    # Correction for InTEM
    if 'sitenames' in var_names:
        var_names.remove('sitenames')
    
    # Scale mole fractions
    for v in var_names:
        ds_model[v] = ds_model[v]/specie_info["mf_units_scaling"]
        ds_model[v].attrs['units'] = specie_info["mf_units_print"]
        
    return ds_model

def slice_flux(ds_all,start_date,end_date,config_data,
               scale_units=True,scale_co2eq=False,convert_flux_units=False,specie=None):
    """
    Slices the flux datasets to within given time limits and 
    scales fluxes into Tg/Gg based on the species.
    
    Args:
        ds_all (dictionary of datasets): 
            xarray datasets read directly from each model's flux netCDF.
        start_date (str): 
            Date to slice data from, e.g. '2021-01-01'
        end_date (str): 
            Date to slice data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filename as keys.
        scale_units (bool): 
            If True, scales country fluxes to Tg or Gy per year.
        scale_co2eq (bool):
            If True, converts country fluxes to CO2-eq in Tg per year.
        convert_flux_units (bool): 
            If True, performs the conversion of molar flux to mass flux (default is False).
        specie (str):
            Gas species, used to choose scaling units, e.g. 'ch4'.
    Returns:
        ds_all (dictionary of datasets):
            xarray datasets, scaled, converted, and sliced between chosen dates.
    
    """
    
    if specie is not None:
        specie_info = config_data['species_info'][specie] 

    #variables that aren't scaled by units
    skip_var = ['flux_total_prior','flux_total_posterior','percentile_flux_total_prior',
                'percentile_flux_total_posterior','countryname','country',
                'country_fraction','outer_region_fraction',
                'covariance_country_flux_total_posterior','flux_total_posterior_inversion_grid']

    for m in ds_all.keys():
        
        m0 = m.split('_')[0]
        
        print(f'\nMasking data from {m}')
        try:
            ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date))
        except:
            ds_all[m] = None
            print(f'No {m} fluxes found between {start_date} and {end_date}')
            print(f'Skipping {m}')
            
        if scale_units == True:
            gwp = 1
            scale_factor = specie_info["units_scaling"][m0]

            # Update scaling factors
            if (scale_co2eq):
                gwp = specie_info["gwp"]
                if (specie_info["units_print"] == "G"): #units_print is expected to be either G or T
                    scale_factor = scale_factor * 1e3 #Convert to Tg
                    # Note: units_print is not re-written because it would go back to
                    #       its original value if initialize_settings is re-run.

            print(f'Scaling {m} country fluxes by {scale_factor*gwp}')
            if ds_all[m] is not None:
                var_names = [k for k in ds_all[m].keys() if k not in skip_var]
                for v in var_names:
                    ds_all[m][v].values = ds_all[m][v].values/scale_factor * gwp

                cov_var = 'covariance_country_flux_total_posterior'
                if cov_var in ds_all[m].keys():
                    ds_all[m][cov_var].values = ds_all[m][cov_var].values/scale_factor**2 * gwp**2
                    print(f'Scaling covariance in {m} by {scale_factor**2 * gwp**2}')
                    
        if convert_flux_units:
            M = specie_info["molar_mass"]
            ds_all[m] = convert_molar_to_mass_flux(ds_all[m], M)
        
    return ds_all


def slice_mf(
        ds_all: dict[str, xr.Dataset],
        config_data: dict[str, dict],
        start_date: str = None,
        end_date: str = None,
        site: str = None,
        baseline_site: str = None,
        data_dir: os.PathLike = None,
        scale_units: bool = False,
        specie: str = None
) -> dict[str, xr.Dataset]:
    """
    Slices down the mole fraction timeseries data, to within the
    given time limits, and/or for the chosen site.
    
    Args:
        ds_all (dictionary of datasets): 
            xarray datasets read directly from each model's flux netCDF.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        start_date (str): 
            Date to slice data from, e.g. '2021-01-01'
        end_date (str): 
            Date to slice data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        site (str):
            Obs site to select data from, e.g. 'MHD'.
        baseline_site (str):
            Site used to define baseline at, options for 'MHD', 'JFJ', or 'CMN'.
            If None, does not mask timeseries by baseline times.
        data_dir (str): 
            Path to top data directory, used to read baseline info files.
        scale_units (bool): 
            If True, scales country fluxes to Tg or Gy per year.
        specie (str):
            Gas specie, used to choose scaling units, e.g. 'ch4'.
    Returns:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates and for 
            chosen site.
    """

    data_dir = Path(data_dir)
    models = list(ds_all.keys())
    specie_info = config_data['species_info'][specie]   

    # Get logical array with baseline timestamps 
    if baseline_site is not None:
        baseline_file = data_dir / 'intem_baseline_timestamps' / f'{baseline_site}_InTEM_baseline_timestamps.nc'

        # Check if files exists
        if not baseline_file.is_file():
            raise FileNotFoundError(f'Cannot find baseline file for masking: {baseline_file}.')

        # Read baseline file
        with xr.open_dataset(baseline_file) as f:
            baseline = f.sel(time=slice(start_date,end_date))
    
    for m in models:
        logger.info(f'Masking data from {m}.')
        
        # Compute offset
        if 'Yav' in ds_all[m].keys():
            offset = int(np.mean(ds_all[m]['Yav'].values))
        else:
            offset = (ds_all[m].time.values[1].astype('datetime64[h]') - ds_all[m].time.values[0].astype('datetime64[h]')).astype(int)

        # Round seconds to integer (correction for elris)
        if 'elris' in m:
            ds_all[m]['time'] = ds_all[m]['time'].dt.round('s')

        # Slice data according to site and time window
        if site is not None:
            site_index = get_site_index(ds_all[m], site)

            if site_index is not None:
                ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date),
                                          nsite=site_index)

                if len(ds_all[m]['time']) == 0:
                    logger.warning(f'No {m} obs found for {site} between {start_date} and {end_date}.')
                    ds_all.pop(m)
                    continue

            else:
                logger.warning(f'No {m} obs found for {site}.')
                ds_all.pop(m)
                continue
        else:
            ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date))

            if len(ds_all[m]['time']) == 0:
                logger.warning(f'No {m} obs found between {start_date} and {end_date}.')
                ds_all.pop(m)
                continue

        # Scale mole fractions
        if scale_units == True and 'flexinvert' not in m:
            ds_all[m] = scale_mf(m, ds_all[m], specie_info)

        # Mask mole fractions according to baseline timestamps
        if baseline_site is not None:
            logger.info('Masking timeseries to only include baseline times.')
                                
            #average baseline mask over obs averaging period
            b = baseline.resample(time=f'{offset}h').mean()
            #adjust baseline mask time back to centre of av period (resample removes this)
            b['time'] = b['time'] + np.timedelta64(offset,'h')/2
                                
            #mask baseline mask again, to only include timestamps where every period in the averaging period is classified as baseline
            b_masked = b.sel(time=b['time'].values[np.where(b['baseline'] == 1.)])
                            
            #mask dataset using only baseline times
            both_times = np.isin(ds_all[m].time.values,b_masked.time.values)          
            ds_all[m] = ds_all[m].sel(time=both_times)
                   
    return ds_all

def get_site_index(ds: xr.Dataset, site: str) -> int | None:

    # Get all sites
    sites = ds['sitenames'].astype(str)

    # Get site index
    if site in sites:
        index = np.where(ds['sitenames'].astype(str) == site)[0][0]
        return index
    
    return None

def get_unique_sites(ds: dict[str, xr.Dataset]):

    sites = []
    for m in ds.keys():
        sites.append(ds[m]['sitenames'].values.astype(str))

    sites = np.sort(np.unique(sites))

    return sites