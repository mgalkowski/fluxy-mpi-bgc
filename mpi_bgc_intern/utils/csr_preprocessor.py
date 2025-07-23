import xarray as xr
import os
import re
import calendar
import numpy as np
import pandas as pd

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
    'flux_land' : ['flux_total_prior', 'flux_total_posterior'],
    'flux' : ['flux_total_prior_country', 'flux_total_posterior_country'],
    'flux_unc' : ['stdev_flux_total_prior_country', 'stdev_flux_total_posterior_country']
}

flux_unit = "mol m-2 s-1"

def preprocess(path_to_prior, path_to_posterior, path_to_output, species):
    ds_prior = xr.open_dataset(path_to_prior)
    ds_posterior = xr.open_dataset(path_to_posterior)

    ds_prior = _rename(ds_prior)
    ds_prior = _convert_time(ds_prior)
    ds_posterior = _rename(ds_posterior)
    ds_posterior = _convert_time(ds_posterior)
    ds = _combine_variable(ds_prior, ds_posterior,
                           species=species,
                           drop_also=[f"{species}flux_ocean", 
                                      f"{species}flux_subt"])

    for var in ds.data_vars:
        if 'rt' in ds[var].dims:
            ds[var] = ds[var].isel(rt=0) 

    if 'rt' in ds.coords:
        ds = ds.drop_vars('rt')
    
    ds = ds.drop_vars(['lproc', 'lrt', 'lspec'])

    _save_dataset(ds, path_to_output)


def _rename(ds):
    rename_dict = {}
    for target_name, possible_names in rename_candidates.items():
        for name in possible_names:
            if name in ds.dims or name in ds.coords or name in ds.data_vars:
                rename_dict[name] = target_name
                break
    return ds.rename(rename_dict) if rename_dict else ds


def _combine_variable(ds_prior, ds_post, species, drop_also=[]):
    ds = ds_prior.copy()
    for v, new_vars in combine_candidates.items():
        varname = f'{species}{v}'
        if varname in ds and varname in ds_post:
            prior_data = ds[varname]
            prior_data = _check_for_units(varname, prior_data, ds)
            post_data = ds_post[varname]
            post_data = _check_for_units(varname, post_data, ds_post)
            
            ds[new_vars[0]] = prior_data
            ds[new_vars[1]] = post_data
            ds = ds.drop_vars([varname])
        else:
            print(f'INFO: {varname} not found in dataset.')

    ds = ds.drop_vars([v for v in drop_also if v in ds])

    return ds


def _check_for_units(varname, data, ds):
     if data.any():
        data.attrs = ds[varname].attrs.copy()
        if 'units' in data.attrs and data.attrs['units'] == 'PgC/yr': 
            area = ds['area'] # Adjust to target region  
            years = _get_years_from_time(ds)
            data = _pgcyr_to_mol_m2_s(data, area, years)
             # Update unit string
            data.attrs['units'] = flux_unit
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
    try:
        ds.to_netcdf(path)
        print(f"✅ File saved successfully: {path}")
    except Exception as e:
        print(f"❌ Error when trying to save file {path}: {e}")





    
        
