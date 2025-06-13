import xarray as xr


rename_candidates = {
    'longitude': ['lon'],
    'latitude' : ['lat'],
    'time' : ['mtime'],
    'country' : ['regname'],
    'area' : ['cell_area']
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
                       new_prior_name="stdevflux_total_prior_country",
                       new_post_name="stdev_flux_total_posterior_country")
    ds.to_netcdf(path_to_output)



def _rename(ds):
    #rename dimensions
    # Dynamically build rename map
    rename_dict = {}

    for target_name, possible_names in rename_candidates.items():
        for name in possible_names:
            if name in ds.dims or name in ds.coords or name in ds.data_vars:
                rename_dict[name] = target_name
                break  # use the first match only
    
    # Apply renaming only if any matches were found
    if rename_dict:
        return ds.rename(rename_dict)
    else: return ds


def _combine_flux_total(ds_prior, ds_posterior):
    # Sum up flux totals for land and ocean and merge prior and posterior fluxes into one file
    
    ds = ds_prior
    flux_to_sum = ['co2flux_land', 'co2flux_ocean']

     # Only sum existing variables
    prior_fluxes = [ds[flux] for flux in flux_to_sum if flux in ds]
    posterior_fluxes = [ds_posterior[flux] for flux in flux_to_sum if flux in ds_posterior]

    # Create new variables only if the sum is not empty
    if prior_fluxes:
        ds['flux_total_prior'] = sum(prior_fluxes)

    if posterior_fluxes:
        ds['flux_total_posterior'] = sum(posterior_fluxes)

    # Drop only variables that exist
    vars_to_drop = [v for v in flux_to_sum if v in ds]
    if 'co2flux_subt' in ds:
        vars_to_drop.append('co2flux_subt')

    if vars_to_drop:
        ds = ds.drop_vars(vars_to_drop)

    return ds

def _combine_flux_total_country(ds_prior, ds_posterior):
    possible_names = {"co2flux", "ch4flux"}
    ds = ds_prior
    for name in possible_names:
        if name in ds.data_vars:
            ds = ds.rename({name: "flux_total_prior_country"})
            break  # use the first match only
    
    for name in possible_names:
        if name in ds_posterior.data_vars:
            ds["flux_total_posterior_country"] = ds_posterior[name]
            break
    return ds


def _combine_stdev_flux_total_country(ds_prior, ds_posterior):
    possible_names = {"co2flux_unc"}
    ds = ds_prior
    for name in possible_names:
        if name in ds.data_vars:
            ds = ds.rename({name: "stdev_flux_total_prior_country"})
            break  # use the first match only
    
    for name in possible_names:
        if name in ds_posterior.data_vars:
            ds["stdev_flux_total_posterior_country"] = ds_posterior[name]
            break
    return ds



def _combine_variable(ds_prior, ds_posterior, varnames, new_prior_name, new_post_name, sum_vars=False, drop_originals=False):
    ds = ds_prior.copy()

    if sum_vars:
        prior_data = [ds[v] for v in varnames if v in ds]
        post_data  = [ds_posterior[v] for v in varnames if v in ds_posterior]

        if prior_data:
            ds[new_prior_name] = sum(prior_data)
        if post_data:
            ds[new_post_name] = sum(post_data)
    else:
        for name in varnames:
            if name in ds.data_vars:
                ds = ds.rename({name: new_prior_name})
                break
        for name in varnames:
            if name in ds_posterior.data_vars:
                ds[new_post_name] = ds_posterior[name]
                break

    if drop_originals:
        varnames.append("co2flux_subt")
        vars_to_drop = [v for v in varnames if v in ds]
        ds = ds.drop_vars(vars_to_drop)
      

    return ds





    