import os
import glob
import xarray as xr
import numpy as np
import json 
import logging

from pathlib import Path
from typing import Literal

from fluxy import config
from fluxy.operators.regions import extract_region_flux
from fluxy.operators.select import slice_flux

logger = logging.getLogger(__name__)

def read_json(filepath: os.PathLike) -> dict:

    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(f'Cannot find {filepath}.')

    with open(filepath, "r") as f:
        json_data = json.load(f)

    return json_data

def read_config_files() -> dict:

    # Get location of json files
    parent_dir = Path(__file__).parent.parent
    configs_dir = parent_dir / 'configs'

    # List of json files to be read
    json_files = configs_dir.glob('*.json')

    # Read json files
    data_dict = {}
    for file in json_files:
        data =  read_json(file)
        filename = file.stem
        data_dict[filename] = data

    return data_dict

def read_model_output(
    data_dir: os.PathLike,
    file_type: Literal["concentration","flux"],
    specie: str,
    models: list[str],
    config_data: dict[str, dict],
    period_override: str | list[str] = None
) -> dict[str, xr.Dataset]:
    """
    Extracts mole fraction or flux timeseries data from each model.
    Args:
        data_dir (str): 
            Path to top data directory.
        specie (str): 
            Gas species, e.g. 'ch4'.
        models (list of str): 
            Keys specifying model names, e.g. ['intem','elris']
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        period_override (list of str) (optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
    Returns:
        ds_all (dictionary of datasets): 
            xarray dataset read directly from each model's mole fraction netCDF.
    """

    specie_info = config_data['species_info'][specie]

    # Set inversion period to default or user defined value
    period_all = {}
    
    if period_override != None and len(period_override) != len(models):
        raise ValueError(f'If using period_override, this list must be of the same length as models.')
    
    for i,m in enumerate(models):
        if period_override is not None:
            if period_override[i] is not None:
                period_all[m] = period_override[i]
            else:
                period_all[m] = specie_info["period"]
        else:
            period_all[m] = specie_info["period"]

    # Define file pattern
    if file_type == 'flux':
        file_pattern = '.nc'
    elif file_type == 'concentration':
        file_pattern = '_concentrations.nc'
    else:
        raise ValueError(f'file_pattern must be equal to "concentration" or "flux".')

    ds_all = {}

    for m in models:
        
        # Get model tag and name
        m0 = m.split('_')[0]
        model_filename = config_data["models_info"][m]["filename"]
        model_dir = model_filename.split('_')[0]
               
        # Define filepath
        data_dir = Path(data_dir) 
        filepath = data_dir / model_dir / specie / f'{model_filename}_{specie_info["model_species"][m0]}_{period_all[m]}{file_pattern}'

        # Check if files exists
        if not filepath.is_file():
            logger.warning(f'Cannot find {file_type} file: {filepath}.')
            continue
 
        # Read file
        logger.info(f'Reading {file_type} file: {filepath}')
        ds_all[m] = xr.open_dataset(filepath)

    return ds_all

def read_flux_total_fgases(data_dir,specie,models,config_data,regions,
                           start_date,end_date,period_override=None,apply_pop_scale=True):
    """
    Reads in fluxes from a list of gases and sums/averages totals and uncertainties,
    to produce one dataset which can be used with plotting functions in the rest 
    of the notebook.

    Args:
        data_dir (str): 
            Path to top data directory.
        specie (str): 
            'all_hfc' or 'all_pfc'
        models (list of str): 
            Keys specifying model names, e.g. ['intem','elris']
        regions (list of str):
            Region names used to extract fluxes. Only these regions can then be plotted.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        start_date (str):
            Date to slice data from, e.g. '2021-01-01'
        end_date (str):
            Date to slice data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        period_override (list of str) (optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
                                       
    Returns:
        ds_all (dictionary of datasets): 
            xarray dataset read directly from each model's flux netCDF.
    """
    
    if specie == 'all_hfc':
        all_species = ['hfc125','hfc134a','hfc143a','hfc152a','hfc23',
                       'hfc227ea','hfc245fa','hfc32','hfc365mfc'] #,'hfc4310mee']
    elif specie == 'all_pfc':
        all_species = ['cf4','pfc116','pfc218','pfc318']
    else:
        raise ValueError('This function can only be used to read total hfc (all_hfc) or total pfc (all_pfc).')
        
       
    if type(start_date) is str:
        print('\nNOTE: Using same start and end date for all models')
        print('If this fails with an error message related to region_time dimensions, check the availablility\n'+
              'of data from all models for all timestamps.\n'+
              'To fix this error, set start_date and end_date as lists with the correct start and end times\nfor each model.')
        start_date = [start_date]*len(models)
        end_date = [end_date]*len(models)

    if period_override == None:
        period_override = [None]*len(all_species)
        
    ds_all = {}
    ds_in = {}
    missing_species = {}

    for m,model in enumerate(models):
        
        #longrun = False
        #if 'longrun' in model:
        #    model = model.split('_')[0]
        #    models[m] = model
        #    longrun = True

        missing_species[model] = []
        m0 = model.split('_')[0]
        
        for s,specie in enumerate(all_species):

            
            #dictionary containing datasets for each species, these are then summed/averaged across the time coordinate
            ds_out = {}

            specie_info = config_data['species_info'][specie]
            
            #tries to read from standard filename
            try:
                model_read = f'{m0}_{specie_info["std_run"][m0]}'
                if 'longrun' in model: model_read = f'{model_read}_longrun'
                
                ds_in[model] = read_model_output(data_dir,"flux",specie,[model_read],config_data,period_override[s])[model_read]
                ds_in[model] = slice_flux(ds_in,start_date[m],end_date[m],config_data,scale_units=False)[model]

            except:
                ds_in[model] = None
                if specie not in missing_species[model]:
                    missing_species[model].append(specie)

            for r,region in enumerate(regions):
                
                try:
                    region_time,region_flux_total_posterior,region_flux_total_prior,\
                    region_flux_total_posterior_lower,region_flux_total_posterior_upper,\
                    region_flux_total_prior_lower,region_flux_total_prior_upper = extract_region_flux(ds_in,model,m0,region,apply_pop_scale,verbose=False)
                    
                    #for percentiles, first convert to upper and lower standard deviations (difference from mean)
                    region_flux_total_posterior_lower = (region_flux_total_posterior-region_flux_total_posterior_lower) * 1e3 * specie_info['gwp'] * 1e-12
                    region_flux_total_posterior_upper = (region_flux_total_posterior_upper-region_flux_total_posterior) * 1e3 * specie_info['gwp'] * 1e-12
                    region_flux_total_prior_lower = (region_flux_total_prior-region_flux_total_prior_lower) * 1e3 * specie_info['gwp'] * 1e-12
                    region_flux_total_prior_upper = (region_flux_total_prior_upper-region_flux_total_prior) * 1e3 * specie_info['gwp'] * 1e-12
                    
                    region_flux_total_posterior = region_flux_total_posterior * 1e3 * specie_info['gwp'] * 1e-12
                    region_flux_total_prior = region_flux_total_prior * 1e3 * specie_info['gwp'] * 1e-12

                    #fix to replace rhime's timestamps, which aren't always in the centre of the inversion period
                    # which breaks the .sum() steps below if trying to include data from missing species
                    if 'rhime' in model:
                        region_time = np.arange(np.datetime64(start_date[m]).astype('datetime64[Y]'),np.datetime64(end_date[m]).astype('datetime64[Y]'),
                                        np.timedelta64(1,'Y')).astype('datetime64[ns]')
                        region_time_extended = np.arange(np.datetime64(start_date[m]).astype('datetime64[Y]'),
                                                        np.datetime64(end_date[m]).astype('datetime64[Y]')+np.timedelta64(1,'Y'),
                                                        np.timedelta64(1,'Y')).astype('datetime64[ns]')
                        time_diff = []
                        for t,test_time in enumerate(region_time):
                            time_diff.append((region_time_extended[t+1] - region_time_extended[t])/2)
                        region_time = region_time + time_diff
                
                except:
                    # create empty set of values for this region and species, so it can be skipped if needed
                    print(f'No {specie} {region} fluxes found for {model} check directory paths and netcdf contents.')
                    region_time = np.arange(np.datetime64(start_date[m]).astype('datetime64[Y]'),np.datetime64(end_date[m]).astype('datetime64[Y]'),
                                            np.timedelta64(1,'Y')).astype('datetime64[ns]')
                    region_time_extended = np.arange(np.datetime64(start_date[m]).astype('datetime64[Y]'),
                                                    np.datetime64(end_date[m]).astype('datetime64[Y]')+np.timedelta64(1,'Y'),
                                                    np.timedelta64(1,'Y')).astype('datetime64[ns]')
                    time_diff = []
                    for t,test_time in enumerate(region_time):
                        time_diff.append((region_time_extended[t+1] - region_time_extended[t])/2)
                    region_time = region_time + time_diff
                    
                    region_flux_total_posterior_lower,region_flux_total_posterior_upper = np.ones(region_time.shape)*np.nan,np.ones(region_time.shape)*np.nan
                    region_flux_total_prior_lower,region_flux_total_prior_upper = np.ones(region_time.shape)*np.nan,np.ones(region_time.shape)*np.nan
                    region_flux_total_posterior,region_flux_total_prior = np.ones(region_time.shape)*np.nan,np.ones(region_time.shape)*np.nan
                 
                    if specie not in missing_species[model]:
                        missing_species[model].append(specie)
                        
                #print(f'{model} {species}')
                #print(region_time)
                 
                if r == 0:
                    country_out = np.array([region])
                    region_flux_total_posterior_out = np.expand_dims(region_flux_total_posterior,axis=1)
                    region_flux_total_prior_out = np.expand_dims(region_flux_total_prior,axis=1)
                    region_flux_total_posterior_lower_out = np.expand_dims(region_flux_total_posterior_lower,axis=1)
                    region_flux_total_posterior_upper_out = np.expand_dims(region_flux_total_posterior_upper,axis=1)
                    region_flux_total_prior_lower_out = np.expand_dims(region_flux_total_prior_lower,axis=1)
                    region_flux_total_prior_upper_out = np.expand_dims(region_flux_total_prior_upper,axis=1)
                else:
                    country_out = np.hstack((country_out,np.array([region])))
                    region_flux_total_posterior_out = np.concatenate((region_flux_total_posterior_out,np.expand_dims(region_flux_total_posterior,axis=1)),axis=1)
                    region_flux_total_prior_out = np.concatenate((region_flux_total_prior_out,np.expand_dims(region_flux_total_prior,axis=1)),axis=1)
                    region_flux_total_posterior_lower_out = np.concatenate((region_flux_total_posterior_lower_out,np.expand_dims(region_flux_total_posterior_lower,axis=1)),axis=1)
                    region_flux_total_posterior_upper_out = np.concatenate((region_flux_total_posterior_upper_out,np.expand_dims(region_flux_total_posterior_upper,axis=1)),axis=1)
                    region_flux_total_prior_lower_out = np.concatenate((region_flux_total_prior_lower_out,np.expand_dims(region_flux_total_prior_lower,axis=1)),axis=1)
                    region_flux_total_prior_upper_out = np.concatenate((region_flux_total_prior_upper_out,np.expand_dims(region_flux_total_prior_upper,axis=1)),axis=1)
                #should be of shape (time,n_country)
                    
            ds_out = xr.Dataset({'region_flux_total_posterior_out':(['region_time','country_out'],region_flux_total_posterior_out),
                                'region_flux_total_prior_out':(['region_time','country_out'],region_flux_total_prior_out),
                                'region_flux_total_posterior_lower_out':(['region_time','country_out'],region_flux_total_posterior_lower_out),
                                'region_flux_total_posterior_upper_out':(['region_time','country_out'],region_flux_total_posterior_upper_out),
                                'region_flux_total_prior_lower_out':(['region_time','country_out'],region_flux_total_prior_lower_out),
                                'region_flux_total_prior_upper_out':(['region_time','country_out'],region_flux_total_prior_upper_out)},
                                coords={'region_time':(['region_time'],region_time),
                                        'country_out':(['country_out'],country_out),
                                        'percentile':(['percentile'],np.array([0.159,0.841]))})
            
            if s == 0:
                ds_out_species_total = ds_out.copy()
            else:
                region_flux_total_posterior_all_species = xr.concat((ds_out_species_total['region_flux_total_posterior_out'],
                                                                    ds_out['region_flux_total_posterior_out']),dim='stack').sum(dim='stack')   #this works when ds_out and ds_out_species_total have different time coordinates
                region_flux_total_prior_all_species = xr.concat((ds_out_species_total['region_flux_total_prior_out'],
                                                                    ds_out['region_flux_total_prior_out']),dim='stack').sum(dim='stack')
                region_flux_total_posterior_lower_all_species = np.sqrt(xr.concat((ds_out_species_total['region_flux_total_posterior_lower_out']**2,
                                                                    ds_out['region_flux_total_posterior_lower_out']**2),dim='stack').sum(dim='stack'))
                region_flux_total_posterior_upper_all_species = np.sqrt(xr.concat((ds_out_species_total['region_flux_total_posterior_upper_out']**2,
                                                                    ds_out['region_flux_total_posterior_upper_out']**2),dim='stack').sum(dim='stack'))
                region_flux_total_prior_lower_all_species = np.sqrt(xr.concat((ds_out_species_total['region_flux_total_prior_lower_out']**2,
                                                                    ds_out['region_flux_total_prior_lower_out']**2),dim='stack').sum(dim='stack'))
                region_flux_total_prior_upper_all_species = np.sqrt(xr.concat((ds_out_species_total['region_flux_total_prior_upper_out']**2,
                                                                    ds_out['region_flux_total_prior_upper_out']**2),dim='stack').sum(dim='stack'))
                
                ds_out_species_total = xr.Dataset({'region_flux_total_posterior_out':(['region_time','country_out'],region_flux_total_posterior_all_species.values),
                                'region_flux_total_prior_out':(['region_time','country_out'],region_flux_total_prior_all_species.values),
                                'region_flux_total_posterior_lower_out':(['region_time','country_out'],region_flux_total_posterior_lower_all_species.values),
                                'region_flux_total_posterior_upper_out':(['region_time','country_out'],region_flux_total_posterior_upper_all_species.values),
                                'region_flux_total_prior_lower_out':(['region_time','country_out'],region_flux_total_prior_lower_all_species.values),
                                'region_flux_total_prior_upper_out':(['region_time','country_out'],region_flux_total_prior_upper_all_species.values)},
                                coords={'region_time':(['region_time'],region_flux_total_prior_all_species['region_time'].values),
                                        'country_out':(['country_out'],country_out)}) 
                
            #print(f"Total = {ds_out_species_total['region_flux_total_posterior_lower_out'].values}")
                
            if m0 == 'intem':
                country_coord_name = 'countrynumber'
            else:
                country_coord_name = 'country'
                
        country_shortnames = []
        for c in ds_out_species_total['country_out'].values:
            try:
                country_shortnames.append(config.countrycodes_dict[c])
            except:
                country_shortnames.append(config.regions_dict_old[c])
                
        ds_all[model] = xr.Dataset({'country_flux_total_prior':(['time',country_coord_name],ds_out_species_total['region_flux_total_prior_out'].values),
                                    'country_flux_total_posterior':(['time',country_coord_name],ds_out_species_total['region_flux_total_posterior_out'].values),
                                    'percentile_country_flux_total_prior':(['time','percentile',country_coord_name],
                                                                        np.concatenate((np.expand_dims((ds_out_species_total['region_flux_total_prior_out'].values-ds_out_species_total['region_flux_total_prior_lower_out'].values),axis=1),
                                                                                    np.expand_dims((ds_out_species_total['region_flux_total_prior_out'].values+ds_out_species_total['region_flux_total_prior_upper_out'].values),axis=1)),axis=1)),
                                    'percentile_country_flux_total_posterior':(['time','percentile',country_coord_name],
                                                                        np.concatenate((np.expand_dims((ds_out_species_total['region_flux_total_posterior_out'].values-ds_out_species_total['region_flux_total_posterior_lower_out'].values),axis=1),
                                                                                    np.expand_dims((ds_out_species_total['region_flux_total_posterior_out'].values+ds_out_species_total['region_flux_total_posterior_upper_out'].values),axis=1)),axis=1))},
                            coords={'time':(['time'],ds_out_species_total['region_time'].values),
                                    country_coord_name:([country_coord_name],np.array(country_shortnames))}) 
    
    missing = []
    for model in models:
        if missing_species[model] != []:
            missing.append(model)
        else:
            print(f'\nAll species succesfully read for {model}!')
            
    for m in missing:
        print(f'\nWARNING: Model {m} is missing species: {missing_species[m]}')

    print('\nTo change the files used as the standard for each HFC/PFC, edit variable std_run in species_info.json')

    return ds_all

def extract_site_info(sites: list[str], config_data: dict[str, dict]):
    """
    Uses info from site_info.json to create a dictionary
    of sites with latitude and longitudes.
    """
        
    site_info = {}
    site_data = config_data['site_info']
    
    for s in sites:
        site_info[s] = {'latitude':site_data[s][list(site_data[s].keys())[0]]['latitude'],
                        'longitude':site_data[s][list(site_data[s].keys())[0]]['longitude']}
    
    return site_info