import xarray as xr

rename_candidates = {
    'longitude': ['lon'],
    'latitude' : ['lat'],
    'time' : ['mtime'],
    'country' : ['regname'],
    'area' : ['cell_area']
}

def preprocessor(prior_path, posterior_path, output_path):

    ds = xr.open_dataset(prior_path)
    ds = rename(ds)
    ds = combine_flux_total(ds, xr.open_dataset(posterior_path))
    output_path = "work/me/processed/flux_data_2021.nc"
    ds.to_netcdf(output_path)



def rename(ds):
    # Dynamically build rename map
    rename_dict = {}

    for target_name, possible_names in rename_candidates.items():
        for name in possible_names:
            if name in ds.dims:
                rename_dict[name] = target_name
                break  # use the first match only
    
    # Apply renaming only if any matches were found
    if rename_dict:
        return ds.rename(rename_dict)
    else: return ds


def combine_flux_total(ds_prior, ds_posterior):
    # Sum up flux totals for land and ocean and merge prior and posterior fluxes into one file
    
    ds = ds_prior
    flux_to_sum = ['co2flux_land', 'co2flux_ocean']
    flux_total_prior = sum(ds[flux] for flux in flux_to_sum)
    ds['flux_total_prior'] = flux_total_prior
    ds = ds.drop_vars(flux_to_sum)
    ds = ds.drop_vars('co2flux_subt')
    flux_total_posterior = sum(ds_posterior[flux] for flux in flux_to_sum)
    ds['flux_total_posterior'] = flux_total_posterior
    return ds