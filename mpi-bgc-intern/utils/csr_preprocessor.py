import xarray as xr
import os
import re

rename_candidates = {
    'longitude': ['lon'],
    'latitude': ['lat'],
    'time': ['mtime'],
    'country': ['regname'],
    'area': ['cell_area']
}


def preprocess(path_to_prior, path_to_posterior, path_to_output):
    ds_prior = xr.open_dataset(path_to_prior)
    ds_posterior = xr.open_dataset(path_to_posterior)

    ds_prior = _rename(ds_prior)
    ds_posterior = _rename(ds_posterior)

    ds = _combine_variable(ds_prior, ds_posterior,
                           varnames=["co2flux_land", "co2flux_ocean"],
                           new_prior_name="flux_total_prior",
                           new_post_name="flux_total_posterior",
                           sum_vars=True,
                           drop_originals=True)

    ds = _combine_variable(ds, ds_posterior,
                           varnames=["co2flux", "ch4flux"],
                           new_prior_name="flux_total_prior_country",
                           new_post_name="flux_total_posterior_country")

    ds = _combine_variable(ds, ds_posterior,
                           varnames=["co2flux_unc"],
                           new_prior_name="stdev_flux_total_prior_country",
                           new_post_name="stdev_flux_total_posterior_country")

    for var in ds.data_vars:
        if 'rt' in ds[var].dims:
            ds[var] = ds[var].isel(rt=0) 

    if 'rt' in ds.coords:
        ds = ds.drop_vars('rt')


    _save_dataset_safely(ds, path_to_output)


def _rename(ds):
    rename_dict = {}
    for target_name, possible_names in rename_candidates.items():
        for name in possible_names:
            if name in ds.dims or name in ds.coords or name in ds.data_vars:
                rename_dict[name] = target_name
                break
    return ds.rename(rename_dict) if rename_dict else ds


def _combine_variable(ds_prior, ds_posterior, varnames, new_prior_name, new_post_name, sum_vars=False, drop_originals=False):
    ds = ds_prior.copy()

    if sum_vars:
        prior_data = [ds[v] for v in varnames if v in ds]
        post_data = [ds_posterior[v] for v in varnames if v in ds_posterior]

        if prior_data:
            combined = sum(prior_data)
            combined.attrs = prior_data[0].attrs.copy()
            if 'units' in combined.attrs and combined.attrs['units'] == 'PgC/yr': 
                area = ds_prior['area'] # Adjust to your target region  
                combined = _pgcyr_to_mol_m2_s(combined, area)
                 # Update unit string
                combined.attrs['units'] = "mol m-2 s-1"
                #combined.attrs['units'] = _convert_unit_fractions(original_unit)
            ds[new_prior_name] = combined

        if post_data:
            combined = sum(post_data)
            combined.attrs = post_data[0].attrs.copy()
            if 'units' in combined.attrs and combined.attrs['units'] == 'PgC/yr':
                area = ds_posterior['area'] # Adjust to your target region  
                combined = _pgcyr_to_mol_m2_s(combined, area)
                 # Update unit string
                combined.attrs['units'] = "mol m-2 s-1"
                #combined.attrs['units'] = _convert_unit_fractions(original_unit)
            ds[new_post_name] = combined
    else:
        for name in varnames:
            if name in ds.data_vars:
                renamed = ds[name]
                renamed.attrs = ds[name].attrs.copy()
                break
        for name in varnames:
            if name in ds_posterior.data_vars:
                ds[new_post_name] = ds_posterior[name]
                ds[new_post_name].attrs = ds_posterior[name].attrs.copy()
                break

    if drop_originals:
        # remove originals (if exist)
        varnames_to_drop = varnames + ["co2flux_subt"]
        ds = ds.drop_vars([v for v in varnames_to_drop if v in ds])

    return ds

def _reduce_to_latlon(ds):
    for var in ds.data_vars:
        v = ds[var]

        # Reduce time dimension
        if "time" in v.dims:
            v = v.mean(dim="time")

        # Reduce rt dimension (retrieval type)
        if "rt" in v.dims:
            v = v.isel(rt=0)

        # Reduce species dimension if it exists
        if "spec" in v.dims:
            v = v.isel(spec=0)

        # Remove singleton dimensions
        v = v.squeeze()

        ds[var] = v
    return ds



def _save_dataset_safely(ds, path):
    try:
        ds.to_netcdf(path)
        print(f"✅ File saved successfully: {path}")
    except Exception as e:
        print(f"❌ Error when trying to save file {path}: {e}")


def _convert_unit_fractions(unit_string):
    """
    Converts unit strings like 'Pg/yr' to 'Pg yr⁻¹', 'kg/m^2/s' to 'kg m⁻2 s⁻¹'.
    """
    if '/' not in unit_string:
        return unit_string  # No fraction to process
    
    parts = unit_string.split('/')
    numerator = parts[0].strip()
    denominators = [p.strip() for p in parts[1:]]
    
    # Replace ^n with superscript negative
    converted_denoms = []
    for d in denominators:
        match = re.match(r'(\w+)\^?(\d*)', d)
        if match:
            unit, power = match.groups()
            power = int(power) if power else 1
            converted_denoms.append(f"{unit}⁻{power}")
        else:
            converted_denoms.append(f"{d}⁻¹")
    
    return f"{numerator} {' '.join(converted_denoms)}"

def _pgcyr_to_mol_m2_s(value_pgcyr, area_m2):
    grams = value_pgcyr * 1e15
    mols = grams / 12.01
    seconds_per_year = 31_536_000
    flux = mols / seconds_per_year
    return flux / area_m2

    
        
