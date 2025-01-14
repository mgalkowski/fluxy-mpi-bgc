import numpy as np
import xarray as xr




def calculate_resample_uncertainty(ds_all_original,ds_all_p,rtime,
                                   resample_uncert_correlation=False):
    """
    Recalculates resampled flux uncertainty, using the assumption
    that all periods in the resampled flux average are uncorrelated.
    Args:
        ds_all_original (dictionary of datasets):
            Extracted flux datasets from flux-format netcdfs.
        ds_all_p (dictionary of datasets):
            Same as above, but resampled down to lower time resolution, 
            using the .mean() method.
        rtime (str):
            Time used for resampling, e.g. 'YS' or 'QS-DEC'.
        resample_uncert_correlation (bool, default False):
            If False, recalculates resampled flux uncertainties with
            uncorrelated assumption. If True, does not recalculate
            uncertainties, and uses the mean uncertainty.
    Returns:
        ds_all_p (dictionary of datasets):
            Same dataset as above, but with updated 'percentile_...'
            terms.
    """
    
    if resample_uncert_correlation == False:
    
        for m in ds_all_original.keys():
            for v in ds_all_original[m].keys():
                if 'percentile_country' in v:
                    n_periods = ds_all_original[m][v].resample(time=rtime).count()[:,0,:]   #number of periods in each average
                    lower = (ds_all_original[m][v.replace('percentile_','')] - ds_all_original[m][v][:,0,:])    #recalculate upper and low standard deviations
                    upper = (ds_all_original[m][v][:,1,:] - ds_all_original[m][v.replace('percentile_','')])
                    lower_resampled = np.sqrt(((lower**2).resample(time=rtime).sum(dim="time")))/n_periods  #resample using sqrt of variances,divided by number of periods
                    upper_resampled = np.sqrt(((upper**2).resample(time=rtime).sum(dim="time")))/n_periods
                    lower_out = ds_all_p[m][v.replace('percentile_','')] - lower_resampled  #recalculated percentile upper and lower bounds
                    upper_out = ds_all_p[m][v.replace('percentile_','')] + upper_resampled
                    
                    ds_all_p[m][v] = xr.DataArray(np.concatenate((np.expand_dims(lower_out,axis=1),
                                                                    np.expand_dims(upper_out,axis=1)),axis=1),
                                                    dims=ds_all_p[m][v].dims)
        
    return ds_all_p
    