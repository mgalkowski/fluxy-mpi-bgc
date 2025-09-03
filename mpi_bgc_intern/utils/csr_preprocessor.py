import xarray as xr
import os
import re
import calendar
import numpy as np
import pandas as pd
import logging

#new_name: ['old_name1', 'old_name2']
rename_candidates = {
    'longitude': ['lon'],
    'latitude': ['lat'],
    'time': ['mtime'],
    'country': ['regname'],
    'area': ['cell_area'],
}
#old_name : ['new_name_prior', 'new_name_posterior']
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

def preprocess(path_to_prior, path_to_posterior, path_to_output, species):
    ds_prior = xr.open_dataset(path_to_prior)
    ds_posterior = xr.open_dataset(path_to_posterior)

    ds_prior = _rename(ds_prior)
    ds_prior = _convert_time(ds_prior)
    ds_prior = _combine_fluxes(ds_prior, species)
    ds_posterior = _rename(ds_posterior)
    ds_posterior = _convert_time(ds_posterior)
    ds_posterior = _combine_fluxes(ds_posterior, species)
    ds = _combine_variable(ds_prior, ds_posterior,
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
                "lspec"
           ]

    to_drop = [v for v in ds.variables if any(sub in v for sub in drop)]

    ds = ds.drop_vars(to_drop)

    prior = ds["flux_total_prior"].values
    posterior = ds["flux_total_posterior"].values
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

def _combine_fluxes(ds_in, species):
    ds = ds_in.copy()
    land = _check_for_units(f"{species}flux_land", ds_in)
    ocean = _check_for_units(f"{species}flux_ocean", ds_in)
    
    flux_combined = land + ocean
    flux_combined.attrs = land.attrs.copy()
    ds[f"{species}{list(combine_candidates.keys())[0]}"] = flux_combined
    
    return ds
    

def _combine_variable(ds_prior, ds_post, species):
    ds = ds_prior.copy()
    for v, new_vars in combine_candidates.items():
        varname = f'{species}{v}'
        if varname in ds and varname in ds_post:
            prior_data = _check_for_units(varname, ds)
            post_data = _check_for_units(varname, ds_post)
            
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

  

def _save_dataset(ds, path):
    ds.encoding.clear()
    for v in ds.variables:
        ds[v].encoding.clear()
    try:
        ds.to_netcdf(path, engine="netcdf4")
        print(f"✅ File saved successfully: {path}")
    except Exception as e:
        print(f"❌ Error when trying to save file {path}: {e}")





    
        
