import pprint
import numpy as np
import xarray as xr

def compute_diff_dataset(ds_all: dict[str, xr.Dataset],
                         models_to_subtract: list[str]
) -> dict[str, xr.Dataset]:

    models = list(ds_all.keys())
        
    if len(models_to_subtract) != 2:
        raise ValueError('List of models to subtract must be of size 2.')
    
    for m in models_to_subtract:
        if m not in models:
            raise KeyError(f'{m} not found in the dataset.')
     
    # Reduce datasets to timestamps/sites common to both models
    ds0, ds1 = xr.align(ds_all[models_to_subtract[0]], ds_all[models_to_subtract[1]], join="inner")

    ds_diff = {}
    key_name = f'{models_to_subtract[0]}-{models_to_subtract[1]}'
    ds_diff[key_name] = ds0

    # Compute difference between the two datasets (excluding non float variables)
    var_names = [k for k in ds_diff[key_name].keys() if k not in ['sitenames','Yav','median_poll_uncert_flag']]
    for v in var_names:
        ds_diff[key_name][v] = ds0[v] - ds1[v]

    return ds_diff

def stats_mf(ds_all):
    """
    Calculates the Pearson correlation coefficent and normalised root
    mean square error, of the fit between the posterior mean mf and the 
    observed mole fraction.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets from slice_mf(), sliced between chosen dates
            but still containing all sites.
    Returns:
        pearson (dictionary of dictionaries):
            Pearson correlation coeffiecient, for each site and for each model.
        nrmse (dictionary of dictionaries):
            Normalised root mean square error, for each site and for each model.
        rmse (dictionary of dictionaries):
            Root mean square error, for each site and for each model (not normalized).
    """
    
    sites_all = np.array([])

    for i,m in enumerate(ds_all.keys()):
        sites_all = np.hstack((sites_all,ds_all[m]['sitenames'].values.astype(str)))
    
    sites_unique,sites_index = np.unique(sites_all,return_index=True)
    sites_all = sites_all[np.sort(sites_index)]
    
    pearson = {}
    nrmse = {}
    rmse = {}
    #std = {}

    for site in sites_all:
        pearson[site] = {}
        nrmse[site] = {}
        rmse[site] = {}
        #std[site] = {}
        for i,m in enumerate(ds_all.keys()):
            if site in ds_all[m]['sitenames'].values.astype('str'):
                s = np.where(ds_all[m]['sitenames'].values.astype('str') == site)[0][0]
                if ds_all[m]['Yobs'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])].shape[0] != 0:
                    pearson[site][m] = np.round(np.corrcoef(ds_all[m]['Yobs'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])],
                                                            ds_all[m]['Yapost'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])])[0,1],3)
                    rmse[site][m] = np.round(np.sqrt(np.mean((ds_all[m]['Yapost'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])]-
                                                    ds_all[m]['Yobs'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])])**2)),3)
                    nrmse[site][m] = np.round(np.sqrt(np.mean((ds_all[m]['Yapost'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])]-
                                                    ds_all[m]['Yobs'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])])**2))/np.mean(ds_all[m]['Yobs'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])]),3)
                    #std[site][m] = np.std(ds_all[m]['Yapost'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])]-
                    #                      ds_all[m]['Yobs'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])])
                    
                else:
                    pearson[site][m] = np.nan
                    nrmse[site][m] = np.nan
                    rmse[site][m] = np.nan
                    #std[site][m] = np.nan
            else:
                pearson[site][m] = np.nan
                nrmse[site][m] = np.nan
                rmse[site][m] = np.nan
                #std[site][m] = np.nan
                
    for site in sites_all:
        if all([np.isnan(v) for v in pearson[site].values()]) == True:
            del pearson[site]
            del nrmse[site]
            del rmse[site]
            #del std[site]
            
    print('\nPearson correlation coefficient:')
    pprint.pprint(pearson,sort_dicts=False)
    
    print('\nNormalised RMSE')
    pprint.pprint(nrmse,sort_dicts=False)

    print('\nRMSE')
    pprint.pprint(rmse,sort_dicts=False)
    
    return pearson,nrmse,rmse
