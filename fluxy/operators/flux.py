import numpy as np
import xarray as xr
import pandas as pd
import calendar
import datetime

def get_flux_mean(
    data: xr.DataArray, 
    season: str = None,
    ) -> xr.DataArray:
    """
    Calculate the mean flux along the 'time' dimension from a dataset, optionally for a specific season.

    Args:
        data (xr.DataArray): 
            The input data containing a 'time' dimension to calculate the mean.
        season (str, optional): 
            The season for which to calculate the mean (e.g., 'DJF', 'MAM', 'JJA', 'SON'). 
            If None, the mean is calculated over the entire 'time' dimension.

    Returns:
        xr.DataArray: 
            The computed mean flux, either over the entire time period or for the specified season.
    """
    if season is None:
        return data.mean(dim="time")

    # Group by season and check if the given season exists
    seasonal_means = data.groupby("time.season", restore_coord_dims=True).mean(dim="time")

    if season not in seasonal_means.season.values:
        raise ValueError(f"Season '{season}' not found in the dataset.")

    return seasonal_means.sel(season=season)

def average_over_dates_list(ds, dates_list):

    # Initialize the groups array to store which group each time value belongs to
    groups = np.full(ds.time.shape, np.nan)

    # Skip the first part of the dataset, starting grouping from dates_list[0]
    # Loop through dates_list and assign each time value to the corresponding group
    for i in range(len(dates_list) - 1):  # We go until the second-to-last element
        groups[(ds.time.values >= dates_list[i]) & (ds.time.values < dates_list[i+1])] = i + 1
    
    # For the last group (from dates_list[-1] to the end), assign remaining times
    groups[ds.time.values >= dates_list[-1]] = len(dates_list)

    groups_da = xr.DataArray(groups, coords={"time": ds.time})

    # Remove the group that corresponds to dates before dates_list[0] (NaN values)
    ds = ds.where(~np.isnan(groups_da), drop=True)
    groups_da = groups_da.where(~np.isnan(groups_da), drop=True)

    ds_avg = ds.groupby(groups_da, restore_coord_dims=True).mean(dim="time")
    ds_avg = ds_avg.rename({'group': 'time'})

    # Create labels based on the first and last date in each group
    time_labels = []
    for group in np.unique(groups_da):
        group_times = ds.time.values[groups_da == group]  # Get times in this group
        first_time = pd.to_datetime(group_times.min())
        last_time = pd.to_datetime(group_times.max())
        time_labels.append(f"{first_time.strftime('%Y/%m')}—{last_time.strftime('%Y/%m')}")

    return ds_avg, time_labels

def average_over_months_list(ds, months_list):

    groups = np.full(ds.time.shape, np.nan)
    for i, months in enumerate(months_list, start=1):
        groups[np.isin(ds.time.dt.month, months)] = i  # Assign group i

    unique_groups = np.unique(groups[~np.isnan(groups)])
    if len(months_list) != len(unique_groups):
        raise ValueError('Some months are missing.')
    groups_da = xr.DataArray(groups, coords={"time": ds.time})
    ds_avg = ds.groupby(groups_da, restore_coord_dims=True).mean(dim="time")
    ds_avg = ds_avg.rename({'group': 'time'})

    time_labels = []
    for i, group in enumerate(unique_groups):
        if isinstance(months_list[i], list):
            time_labels.append("—".join(calendar.month_abbr[m] for m in months_list[i]))
        else:
            time_labels.append(calendar.month_abbr[months_list[i]])

    return ds_avg, time_labels

def average_over_seasons(ds):

    groups = ds.time.dt.season
    ds_avg = ds.groupby(groups, restore_coord_dims=True).mean(dim="time")
    ds_avg = ds_avg.rename({'season': 'time'})
    time_labels = np.unique(groups.values).tolist()
    return ds_avg, time_labels

def average_over_years(ds, N=1):

    groups = ds.time.dt.year // N
    ds_avg = ds.groupby(groups, restore_coord_dims=True).mean(dim="time")
    ds_avg = ds_avg.rename({'year': "time"})

    first_time_groups = ds.time.groupby(groups, restore_coord_dims=True).first()
    last_time_groups = ds.time.groupby(groups, restore_coord_dims=True).last()

    ds_avg['time'] = first_time_groups.values

    time_labels = []
    for i, group in enumerate(np.unique(groups.values)):
        first_time = first_time_groups[i].dt.strftime('%Y').astype(str).values
        end_time = last_time_groups[i].dt.strftime('%Y').astype(str).values

        if first_time == end_time:
            time_labels.append(f"{first_time}")
        else:
            time_labels.append(f"{first_time}—{end_time}")
    return ds_avg, time_labels

def average_over_months(ds, N=1):

    groups = (ds.time.dt.year * 12 + ds.time.dt.month - 1) // N
    ds_avg = ds.groupby(groups, restore_coord_dims=True).mean(dim="time")
    ds_avg = ds_avg.rename({'group': "time"})

    first_time_groups = ds.time.groupby(groups, restore_coord_dims=True).first()
    last_time_groups = ds.time.groupby(groups, restore_coord_dims=True).last()

    ds_avg['time'] = first_time_groups.values

    time_labels = []
    for i, group in enumerate(np.unique(groups.values)):
        first_time = first_time_groups[i].dt.strftime('%Y-%m').astype(str).values
        end_time = last_time_groups[i].dt.strftime('%Y-%m').astype(str).values

        if first_time == end_time:
            time_labels.append(f"{first_time}")
        else:
            time_labels.append(f"{first_time}—{end_time}")
    return ds_avg, time_labels

def average_over_period(ds, N, chop_by='year'):

    if isinstance(chop_by, list):
        # Case where chop_by is a list of dates
        if all(isinstance(i, (str, datetime.date, np.datetime64, pd.Timestamp)) for i in chop_by):
            dates_list = np.array([np.datetime64(pd.to_datetime(date), 'ns') for date in chop_by]) 
            return average_over_dates_list(ds.copy(), dates_list)
        
        # Case where chop_by is either a list of lists or a list of numbers.
        if all(isinstance(i, (list, int, float)) for i in chop_by):
            months_list = chop_by
            return average_over_months_list(ds.copy(), months_list)

    elif chop_by == 'season':
        return average_over_seasons(ds.copy())
    elif chop_by == 'year':
        return average_over_years(ds.copy(), N)
    elif chop_by == 'month':
        #TODO Check if monthly inversions
        return average_over_months(ds.copy(), N)
    else:
        raise ValueError(f'Option {chop_by} for chop_by not implemented. Options are year, month, season, a list of starting dates or a list of month numbers.')

def define_var_plot(ds: xr.Dataset, var: str):

    if var == 'posterior_prior_diff':
        var_plot = ds['flux_total_posterior'] - ds['flux_total_prior']
    elif var == 'posterior_mean_diff':
        var_plot = ds['flux_total_posterior'] - ds['flux_total_posterior'].mean(dim='time')
    elif var == 'posterior_prior_diff_inversion_grid':
        var_plot = ds['flux_total_posterior_inversion_grid'] - ds['flux_total_prior']
    elif var == 'posterior_mean_diff_inversion_grid':
        var_plot = ds['flux_total_posterior_inversion_grid'] - ds['flux_total_posterior_inversion_grid'].mean(dim='time')
    else:
        if var not in ds:
            raise ValueError(f"'{var}' not found in dataset(s)")
        var_plot = ds[var]

    return var_plot