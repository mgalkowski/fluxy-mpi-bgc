import xarray as xr
import numpy as np
import os


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


def slice_mf(ds_all,config_data,start_date=None,end_date=None,site=None,
             baseline_site=None,data_dir=None,
             scale_units=False,
             species=None):
    """
    Slices down the mole fraction timeseries data, to within the
    given time limits, and/or for the chosen site.
    
    Args:
        ds_all (dictionary of datasets): 
            xarray datasets read directly from each model's flux netCDF.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
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
        species (str):
            Gas species, used to choose scaling units, e.g. 'ch4'.
    Returns:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates and for 
            chosen site.
    """
    
    species_info = config_data['species_info'][species]

    if baseline_site is not None:
        with xr.open_dataset(os.path.join(data_dir,f'intem_baseline_timestamps/{baseline_site}_InTEM_baseline_timestamps.nc')) as f:
            baseline = f.sel(time=slice(start_date,end_date))
    
    for m in ds_all.keys():
        print(f'\nMasking data from {m}')
        
        if 'Yav' in ds_all[m].keys():
            offset = int(np.mean(ds_all[m]['Yav'].values))
        else:
            offset = (ds_all[m].time.values[1].astype('datetime64[h]') - ds_all[m].time.values[0].astype('datetime64[h]')).astype(int)

        # round seconds to integer (correction for elris)
        if 'elris' in m:
            ds_all[m]['time'] = ds_all[m]['time'].dt.round('s')

        if site is not None:
            try:
                site_index = np.where(ds_all[m]['sitenames'].astype(str) == site)[0][0]
                ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date),
                                        nsite=site_index)
            except:
                ds_all[m] = None
                print(f'No {m} obs found for {site} between {start_date} and {end_date}')
        else:
            try:
                ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date))
            except:
                ds_all[m] = None
                print(f'No {m} obs found between {start_date} and {end_date}')
                
        if scale_units == True:
            if 'flexinvert' in m:
                print('No scaling for flexinvert')
            else:
                print(f'Scaling {m} units by {species_info["mf_units_scaling"]}')
                if ds_all[m] is not None:
                    var_names = [k for k in ds_all[m].keys() if k not in ['sitenames','Yav','median_poll_uncert_flag']]
                    for v in var_names:
                        ds_all[m][v] = ds_all[m][v]/species_info["mf_units_scaling"]
      
        if baseline_site is not None:
            print('Masking timeseries to only include baseline times')
            
            try:
                                        
                #average baseline mask over obs averaging period
                b = baseline.resample(time=f'{offset}H').mean()
                #adjust baseline mask time back to centre of av period (resample removes this)
                b['time'] = b['time'] + np.timedelta64(offset,'h')/2
                                    
                #mask baseline mask again, to only include timestamps where every period in the averaging period is classified as baseline
                b_masked = b.sel(time=b['time'].values[np.where(b['baseline'] == 1.)])
                                
                #mask dataset using only baseline times
                both_times = np.isin(ds_all[m].time.values,b_masked.time.values)
                                
                ds_all[m] = ds_all[m].sel(time=both_times)
                    
            except:
                print('Failed to mask {m} data by baseline times')
    
    check_keys = list(ds_all.keys())
    for m in check_keys:
        if ds_all[m] is None:
            ds_all.pop(m)
                
    return ds_all
