import xarray as xr
import os

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
            ds[new_prior_name] = combined

        if post_data:
            combined = sum(post_data)
            combined.attrs = post_data[0].attrs.copy()
            ds[new_post_name] = combined
    else:
        for name in varnames:
            if name in ds.data_vars:
                renamed = ds[name]
                renamed.attrs = ds[name].attrs.copy()
                ds = ds.rename({name: new_prior_name})
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


def _save_dataset_safely(ds, path):
    try:
        ds.to_netcdf(path)
        print(f"✅ Datei erfolgreich gespeichert: {path}")
    
        
