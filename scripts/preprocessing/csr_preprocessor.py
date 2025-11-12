import xarray as xr
import os
import re
import calendar
import numpy as np
import pandas as pd
import logging
from pathlib import Path

rename_candidates = {
    'longitude': ['lon'],
    'latitude': ['lat'],
    'time': ['mtime'],
    'country': ['regname'],
    'area': ['cell_area'],
}

combine_candidates = {
    'flux_combined' : ['flux_total_prior', 'flux_total_posterior'],
    'flux' : ['flux_total_prior_country', 'flux_total_posterior_country'],
    'flux_unc' : ['stdev_flux_total_prior_country', 'stdev_flux_total_posterior_country']
}

unit_conversions = {
        "PgC/yr": 1,
        "TgC/yr": 1/1000,
        "Tmol/yr": 12.01/1000
    }

flux_unit = "mol m-2 s-1"

country_code_EUROPE30f = ['none','AUT','BEL','BGR','CHE','CYP','CZE','DEU','DNK','EST','GRC','ESP','FIN','FRA','HRV','HUN','IRL',
                          'ITA','LTU','LUX','LVA','MLT','NLD','POL','PRT','ROU','SWE','SVN','SVK','GBR','LAND','OCEAN']

country_code_FLUXYf = ['CHE','SWE','ESP','SVK','SVN','PRT','POL','NOR','LUX','ITA','IRL','HUN','DEU','CZE','HRV','BEL','AUT','OCEAN']

country_code_FLUXYALLf = ['CHE','SWE','ESP','SVK','SVN','ROU','PRT','POL','NOR','MLT','LUX','LTU','LVA','ITA','IRL','HUN','GRC','DEU',
                          'EST','CZE','CYP','HRV','BGR','BEL','AUT','TUR','WEE','CEE','NOE','SOE','SEE','EAE','BNL','UKI','IBE','E28',
                          'E27','E15','EUR','LAND','OCEAN']

header_names_rhs = ['time','row','col','covar_pri','covar_post','corr_pri','corr_post','deriv_mu']

def preprocess(path_to_prior, path_to_posterior, path_to_prior_country, path_to_posterior_country, path_to_uncertainty_country,
               path_to_output, species):

    # --- combine gridded and country-aggregated prior and posterior fluxes ---
    
    ds_prior = xr.open_dataset(path_to_prior,decode_timedelta=True)
    ds_posterior = xr.open_dataset(path_to_posterior,decode_timedelta=True)
    ds_prior_country = xr.open_dataset(path_to_prior_country,decode_timedelta=True)
    ds_posterior_country = xr.open_dataset(path_to_posterior_country,decode_timedelta=True) 

    ds_prior = _rename(ds_prior)
    ds_prior = _convert_time(ds_prior)
    ds_prior = _combine_fluxes(ds_prior, species)
    ds_posterior = _rename(ds_posterior)
    ds_posterior = _convert_time(ds_posterior)
    ds_posterior = _combine_fluxes(ds_posterior, species)
    ds_prior_country = _rename(ds_prior_country) 
    ds_prior_country = _rename_country_id(ds_prior_country) 
    ds_prior_country = _convert_time(ds_prior_country) 
    ds_prior_country = _combine_fluxes_country(ds_prior_country, species)
    ds_posterior_country = _rename(ds_posterior_country) 
    ds_posterior_country = _rename_country_id(ds_posterior_country) 
    ds_posterior_country = _convert_time(ds_posterior_country) 
    ds_posterior_country = _combine_fluxes_country(ds_posterior_country, species)

    ds = _combine_variable(ds_prior, ds_posterior, ds_prior_country, ds_posterior_country,
                           species=species)

    for var in ds.data_vars:
        if 'rt' in ds[var].dims:
            ds[var] = ds[var].isel(rt=0) 

    if 'rt' in ds.coords:
        ds = ds.drop_vars('rt')
    
    drop = ["flux_land",
                "flux_ocean", 
                "flux_subt",
                "flux_excl",
                "lproc",
                "lrt", 
                "lspec",
                "dt" # remove to avoid encoding issues (anyway not needed)
           ]

    to_drop = [v for v in ds.variables if any(sub in v for sub in drop)]

    ds = ds.drop_vars(to_drop)

    prior = ds["flux_total_prior"].values
    posterior = ds["flux_total_posterior"].values

    # --- add uncertainties for country-aggregated fluxes from RHS-runs ---

    if isinstance(path_to_uncertainty_country, list):
        # check if RHS results are available for each flux time-step (e.g. each year or month) 
        if len(ds['time'])!=len(path_to_uncertainty_country):
            print("Number of RHS results and number of time steps do not match! Stop RHS processing.")
        else:
            data_prior = np.zeros((len(ds['country']),len(ds['time'])))
            data_posterior = np.zeros((len(ds['country']),len(ds['time'])))
            for i in range(0,len(path_to_uncertainty_country)):
                if os.path.exists(path_to_uncertainty_country[i]): 
                    # read RHS results for each flux time-step
                    df_cross = pd.read_csv(path_to_uncertainty_country[i],comment='#',sep=' ',names=header_names_rhs, skipinitialspace=True)
                    unc_prior_country_file = np.sqrt(df_cross.loc[df_cross['row'] == df_cross['col'], 'covar_pri'])
                    unc_posterior_country_file = np.sqrt(df_cross.loc[df_cross['row'] == df_cross['col'], 'covar_post'])
                    unc_prior_country_file = _tmolyr_to_kg_s_ch4_unc_rhs(unc_prior_country_file,ds['time'][i])
                    unc_posterior_country_file = _tmolyr_to_kg_s_ch4_unc_rhs(unc_posterior_country_file,ds['time'][i]) 
                else:
                    unc_prior_country_file = np.nan
                    unc_posterior_country_file = np.nan
                data_prior[:,i] = unc_prior_country_file 
                data_posterior[:,i] = unc_posterior_country_file 

            # write to xarray:
            unc_prior = xr.DataArray(
                data=data_prior,
                coords={'country': ds['country'].values, 'time': ds['time'].values},
                dims=['country','time'],
                name='stdev_flux_total_prior_country' 
            )
            unc_prior.attrs['units'] = 'kg s-1'

            unc_posterior = xr.DataArray(
                data=data_posterior,
                coords={'country': ds['country'].values, 'time': ds['time'].values},
                dims=['country','time'],
                name='stdev_flux_total_posterior_country' 
            )
            unc_posterior.attrs['units'] = 'kg s-1'

            ds['stdev_flux_total_prior_country'] = unc_prior 
            ds['stdev_flux_total_posterior_country'] = unc_posterior 

    else: 
        print("Uncertainties for country-aggregated fluxes are not available.")
  
    print("_save_dataset")    
    
    _save_dataset(ds, path_to_output)


def _rename(ds):
    rename_dict = {}
    for target_name, possible_names in rename_candidates.items():
        for name in possible_names:
            if name in ds.dims or name in ds.coords or name in ds.data_vars:
                rename_dict[name] = target_name
                break
    return ds.rename(rename_dict) if rename_dict else ds

def _rename_country_id(ds_in): 
    ds = ds_in.copy()
    if "EUROPE30f" in ds.attrs['filename']:
        ds['country'] = (("reg"), country_code_EUROPE30f)
    if "FLUXYf" in ds.attrs['filename']:
        ds['country'] = (("reg"), country_code_FLUXYf)
    if "FLUXYALLf" in ds.attrs['filename']:
        ds['country'] = (("reg"), country_code_FLUXYALLf)
    return ds

def _combine_fluxes(ds_in, species):
    ds = ds_in.copy()
    land = _check_for_units(f"{species}flux_land", ds_in)
    ocean = _check_for_units(f"{species}flux_ocean", ds_in)

    if land is None and ocean is None:
        print("DEBUG: Only combined flux available!")
        # Fallback to combined flux if neither land nor ocean is available
        combined = _check_for_units(f"{species}flux", ds_in)
        if combined is None:
            raise ValueError(
                f"No {species}flux_land, {species}flux_ocean, or {species}flux in dataset"
            )
        flux_combined = combined
        attrs = _copy_attrs(combined)
    elif land is None:
        flux_combined = ocean
        attrs = _copy_attrs(ocean)
    elif ocean is None:
        flux_combined = land
        attrs = _copy_attrs(land)
    else:
        flux_combined = land + ocean
        attrs = _copy_attrs(land)
        attrs.update(_copy_attrs(ocean))

    flux_combined.attrs = attrs
        
    ds[f"{species}{list(combine_candidates.keys())[0]}"] = flux_combined
    
    return ds

def _combine_fluxes_country(ds_in, species):
    ds = ds_in.copy()
    flux = _check_for_units_country(f"{species}flux", ds_in)

    ds[f"{species}{list(combine_candidates.keys())[1]}"] = flux
    
    return ds

def _copy_attrs(var, default=None):
    return var.attrs.copy() if hasattr(var, "attrs") else (default or {})
    

def _combine_variable(ds_prior, ds_post, ds_prior_country, ds_post_country, species):
    ds = ds_prior.copy()
    for v, new_vars in combine_candidates.items():
        varname = f'{species}{v}'
        if varname in ds and varname in ds_post:
            prior_data = _check_for_units(varname, ds)
            post_data = _check_for_units(varname, ds_post)
            
            ds[new_vars[0]] = prior_data
            ds[new_vars[1]] = post_data
            ds = ds.drop_vars([varname])
            
        if varname in ds_prior_country and varname in ds_post_country:
            prior_data_country = _check_for_units_country(varname, ds_prior_country)
            post_data_country = _check_for_units_country(varname, ds_post_country)
            
            # (reg,time) -> (country,time)
            country = ds_prior_country['country']
            prior_data_country.coords['reg'] = country
            prior_data_country = prior_data_country.rename({'reg': 'country'})
            post_data_country.coords['reg'] = country
            post_data_country = post_data_country.rename({'reg': 'country'})
            
            ds[new_vars[0]] = prior_data_country
            ds[new_vars[1]] = post_data_country
            #ds = ds.drop_vars([varname]) #'ch4flux' not in gridded prior file
        
        else:
            print(f'INFO: {varname} not found in dataset.')

    return ds

def _combine_variable_country(ds_prior_country, ds_post_country, species):
    ds = ds_prior.copy()
    for v, new_vars in combine_candidates.items():
        varname = f'{species}{v}'
        if varname in ds_prior_country and varname in ds_post_country:
            prior_data = _check_for_units(varname, ds_prior_country)
            post_data = _check_for_units(varname, ds_post_country)
            
            ds[new_vars[0]] = prior_data
            ds[new_vars[1]] = post_data
            ds = ds.drop_vars([varname])
        else:
            print(f'INFO: {varname} not found in dataset.')

    return ds


def _check_for_units(varname, ds):
    if varname not in ds:
        logging.warning(f"Variable {varname} not found in dataset.")
        return None

    data = ds[varname]
    if data.size == 0 or np.all(np.isnan(data)):
        logging.warning(f"Variable {varname} is empty or all NaN.")
        return None

    # Copy attrs to preserve metadata
    data.attrs = ds[varname].attrs.copy()

    if 'units' in data.attrs:
        unit = data.attrs['units']
        if "dxyp" not in ds:
            raise KeyError("Dataset is missing 'dxyp' (grid cell area), required for flux conversion.")

        area = ds["dxyp"]
        years = _get_years_from_time(ds)

        if unit in unit_conversions:
            data = data * unit_conversions[unit]
            data = _pgcyr_to_mol_m2_s(data, area, years)
            data.attrs['units'] = "mol m-2 s-1"
        else:
            logging.info(f"No conversion applied for variable {varname} with unit '{unit}'.")

    return data

def _check_for_units_country(varname, ds): #f"{species}flux"
    if varname not in ds:
        logging.warning(f"Variable {varname} not found in dataset.")
        return None

    data = ds[varname]
    if data.size == 0 or np.all(np.isnan(data)):
        logging.warning(f"Variable {varname} is empty or all NaN.")
        return None

    # Copy attrs to preserve metadata
    data.attrs = ds[varname].attrs.copy()

    if 'units' in data.attrs:
        unit = data.attrs['units']

        years = _get_years_from_time(ds)

        if unit in unit_conversions:
            data = data * unit_conversions[unit]
            data = _pgcyr_to_kg_s_ch4(data, years)
            data.attrs['units'] = "kg s-1"
        else:
            logging.info(f"No conversion applied for variable {varname} with unit '{unit}'.")

    return data
         
def _pgcyr_to_mol_m2_s(value_pgcyr, area_m2, years):
    
    grams = value_pgcyr * 1e15
    mols = grams / 12.01

    days_in_year = np.array([366 if calendar.isleap(y) else 365 for y in years])
    seconds_per_year = days_in_year * 24 * 60 * 60

      # Make this an xarray DataArray to allow broadcasting
    seconds_per_year = xr.DataArray(
        seconds_per_year,
        dims=["time"],
        coords={"time": value_pgcyr["time"]}
    )
   
    flux = mols / seconds_per_year
    return flux / area_m2

def _pgcyr_to_kg_s_ch4(value_pgcyr, years):
    
    kilograms = value_pgcyr * 1e12
    kilograms_ch4 = kilograms / 12.01 * 16.04

    days_in_year = np.array([366 if calendar.isleap(y) else 365 for y in years])
    seconds_per_year = days_in_year * 24 * 60 * 60

    # Make this an xarray DataArray to allow broadcasting
    seconds_per_year = xr.DataArray(
        seconds_per_year,
        dims=["time"],
        coords={"time": value_pgcyr["time"]}
    )
   
    flux = kilograms_ch4 / seconds_per_year
    return flux 

def _convert_time(ds):
    if np.issubdtype(ds['time'].dtype, np.datetime64):
        # already datetime64 — no need to convert
        return ds

    # convert seconds since 2000-01-01 to datetime64
    base = np.datetime64("2000-01-01T00:00:00", "s")
    time_seconds = ds['time'].values.astype("timedelta64[s]")
    new_time = base + time_seconds

    # set the new time coordinate, forcing ns resolution to avoid warnings
    ds = ds.assign_coords(time=new_time.astype("datetime64[ns]"))
    return ds

def _get_years_from_time(ds):
    # Create datetime64 array for time coordinate (days since 1970-01-01)
    dates = np.datetime64('1970-01-01') + ds['time'].values.astype('timedelta64[D]')

    # Extract years as integers
    years = dates.astype('datetime64[Y]').astype(int) + 1970

    return years

def _tmolyr_to_kg_s_ch4_unc_rhs(unc_rhs, time_rhs):
    # seconds per year 
    year = pd.to_datetime(time_rhs.values).year
    days_in_year = np.array([366 if calendar.isleap(year) else 365])
    seconds_per_year = days_in_year * 24 * 60 * 60
    # convert Tmol -> kgCH4/s
    unc_rhs = unc_rhs * 16.04 * 1e9/seconds_per_year

    return unc_rhs

  

def _save_dataset(ds, path):
    ds.encoding.clear()
    for v in ds.variables:
        ds[v].encoding.clear()
    try:
        if os.path.exists(path):
            os.remove(path)
        # make sure parent folders exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path, engine="netcdf4")
        print(f"✅ File saved successfully: {path}")
    except Exception as e:
        print(f"❌ Error when trying to save file {path}: {e}")





    
        
