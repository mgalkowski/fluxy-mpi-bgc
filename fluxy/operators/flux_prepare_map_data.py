import xarray as xr
from fluxy.operators.flux_align_dataset import (
    align_time,
    align_lat_lon,
)


def define_var_plot(
    ds: xr.Dataset,
    var: str,
) -> xr.DataArray:
    """
    Define the variable to be plotted based on the specified `var` string.

    This function returns the appropriate variable from the dataset `ds` based on
    the value of `var`. It supports predefined differences between flux variables
    or directly returns a variable from the dataset.

    Args:
        ds (xarray.Dataset):
            The input dataset containing various flux variables.
        var (str):
            The variable name or difference type to be plotted. Options for difference include:
                   'posterior_prior_diff', 'posterior_mean_diff',
                   'posterior_prior_diff_inversion_grid', 'posterior_mean_diff_inversion_grid'.

    Returns:
        var_plot (xarray.DataArray):
            The variable or computed difference to be plotted.
    """

    if var == "posterior_prior_diff":
        var_plot = ds["flux_total_posterior"] - ds["flux_total_prior"]
    elif var == "posterior_mean_diff":
        var_plot = ds["flux_total_posterior"] - ds["flux_total_posterior"].mean(
            dim="time"
        )
    elif var == "posterior_prior_diff_inversion_grid":
        var_plot = ds["flux_total_posterior_inversion_grid"] - ds["flux_total_prior"]
    elif var == "posterior_mean_diff_inversion_grid":
        var_plot = ds["flux_total_posterior_inversion_grid"] - ds[
            "flux_total_posterior_inversion_grid"
        ].mean(dim="time")
    else:
        if var not in ds:
            raise ValueError(f"'{var}' not found in dataset(s)")
        var_plot = ds[var]

    return var_plot


def prepare_data(
    ds_all: dict[xr.Dataset], plot_combined: bool = False
) -> dict[xr.Dataset]:
    """
    Prepare flux datasets for flux maps by filtering variables, removing unused dimensions, and aligning time and spatial coordinates.

    Args:
        ds_all (dict[xr.Dataset]):
            Dictionary of model names and corresponding xarray datasets.
        plot_combined (bool, optional):
            If True, returns a single averaged dataset.

    Returns:
        ds_dict (dict[xr.Dataset]):
            Processed datasets, either individually or as a combined average.
    """

    for key, ds in ds_all.items():
        # Step 1: Remove variables without 'time', 'latitude' and 'longitude'
        ds = ds.drop_vars(
            [
                var
                for var in ds.data_vars
                if not {"time", "latitude", "longitude"}.issubset(ds[var].dims)
            ]
        )
        # Step 2: Remove unused coordinates (dimensions that are no longer used)
        unused_dims = set(ds.dims) - set(
            dim for var_i in ds.data_vars for dim in ds[var_i].dims
        )
        ds_all[key] = ds.drop_dims(unused_dims)

    # Align dataset coordinates
    models = list(ds_all.keys())
    ds_list = list(ds_all.values())
    ds_list = align_time(ds_list)
    ds_list = align_lat_lon(ds_list, coord="latitude")
    ds_list = align_lat_lon(ds_list, coord="longitude")

    if plot_combined:
        ds_dict = {
            "combined": xr.concat(ds_list, dim="models", combine_attrs="override").mean(
                dim="models", keep_attrs=True
            )
        }
    else:
        ds_dict = dict(zip(models, ds_list))

    return ds_dict
