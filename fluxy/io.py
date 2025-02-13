import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import json 
import geopandas as gpd
import logging

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from pathlib import Path
from typing import Literal

from fluxy import config
from fluxy.operators.regions import extract_region_flux
from fluxy.operators.select import slice_flux
from fluxy.operators.flux_align_dataset import align_dataset

logger = logging.getLogger(__name__)

def read_json(filepath: os.PathLike) -> dict[str, dict]:
    """
    Reads json file.

    Args:
        filepath (str or Path):
            Path to json file including filename.
    Returns:
        json_data (dictionary of dictionaries):
            Dictionary with data read from filepath.
    """

    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(f'Cannot find {filepath}.')

    with open(filepath, "r") as f:
        json_data = json.load(f)

    return json_data

def read_config_files() -> dict[str, dict]:
    """
    Reads all configuration json files.

    Returns:
        data_dict (dictionary of dictionaries):
            Dictionary with keys equal to json basename (without extension).
            Each key points to a dictionary with the data from each json file.
    """

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

def get_filename(model, species, period, file_pattern, config_data, data_dir):

    # Get file name tags
    name_tags = model.split('_')
    model_name = name_tags[0]
    param_tags = name_tags[3:]

    # Replace parameter tags by dict values in config
    filename_tags = config_data["models_info"].get("filename_tags", None)
    if filename_tags is not None:
        for i,param in enumerate(param_tags):
            string_in_file = filename_tags.get(param, None)

            if string_in_file is not None:
                string_in_file = string_in_file.replace("<model>", model_name.lower())
                name_tags[i+3] = string_in_file
                
    # Add domain to filename tags
    name_tags.insert(2, config_data["models_info"]["domain"])

    # Build filename
    model_filename = "_".join(name_tags)

    # Get species name
    species_print = species
    if (species_names := config_data["models_info"].get("species_name")) and \
       (model_species := species_names.get(model_name)) and \
       (species_tag   := model_species.get(species)):
        species_print = species_tag
            
    # Define filepath
    data_dir = Path(data_dir)
    filepath = data_dir / model_name / species / f'{model_filename}_{species_print}_{period}{file_pattern}'
    
    return filepath

def read_model_output(
    data_dir: os.PathLike,
    file_type: Literal["concentration","flux"],
    species: str,
    models: list[str],
    config_data: dict[str, dict],
    period: str | list[str] = 'yearly',
) -> dict[str, xr.Dataset]:
    """
    Extracts mole fraction or flux timeseries data from each model.

    Args:
        data_dir (str): 
            Path to top data directory.
        species (str): 
            Gas species, e.g. 'ch4'.
        models (list of str): 
            Keys specifying model names, e.g. ['intem','elris']
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        period (str or list of str):
            Inversion period as specified in the model filename.
            If it is a string, the same period is considered for all models.
            If it is a list, one value per model must be specified, e.g. ['monthly','yearly']
    Returns:
        ds_all (dictionary of datasets): 
            xarray dataset read directly from each model's mole fraction netCDF.
    """
    
    if isinstance(period, str):
        period = [period]*len(models)
    
    if len(period) != len(models):
        raise ValueError(f'period must be a string or a list of the same length as models.')

    # Define file pattern
    if file_type == 'flux':
        file_pattern = '.nc'
    elif file_type == 'concentration':
        file_pattern = '_concentrations.nc'
    else:
        raise ValueError(f'file_pattern must be equal to "concentration" or "flux".')

    ds_all = {}

    for i,m in enumerate(models):
        filepath = get_filename(m, species, period[i], file_pattern, config_data, data_dir)

        # Check if files exists
        if not filepath.is_file():
            logger.warning(f'Cannot find {file_type} file: {filepath}.')
            continue
 
        # Read file
        logger.info(f'Reading {file_type} file: {filepath}')
        ds_all[m] = xr.open_dataset(filepath)

        # Add/correct attributes
        ds_all[m] = edit_ds(ds_all[m],m,period[i],file_type)

    return ds_all

def read_flux_total_fgases(data_dir: str, 
    species: str, 
    models: str | list, 
    config_data: dict[str, dict], 
    regions: list |str,
    start_date: str,end_date: str,
    period: str = 'yearly',
    run_keys: str | list = 'default'
) -> dict[str, xr.Dataset]:
    """
    Reads in fluxes from a list of gases and sums/averages totals and uncertainties,
    to produce one dataset which can be used with plotting functions in the rest 
    of the notebook.

    Args:
        data_dir (str): 
            Path to top data directory.
        species (str): 
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
        period (str or list of str):
            Inversion period as specified in the model filename.
            If it is a string, the same period is considered for all models.
            If it is a list, one value per model must be specified, e.g. ['monthly','yearly']
        run_keys : 
            keys to find the runs to use in the config file
                 
    Returns:
        ds_all (dictionary of datasets): 
            xarray dataset read directly from each model's flux netCDF.
    """
    logger.warning(" /!!!!!\ start_date and end_date as lists won't work for now /!!!!!\ ")

    all_species = config_data['species_info'].get(species,{'list_species':None}).get('list_species',None)
    if all_species is None:
        raise ValueError(f'No list of species was found in the config_data for {species}. Please update your config file.')
        
    if type(start_date) is str:
        logger.warning(' Using same start and end date for all models')
        logger.info('If this fails with an error message related to region_time dimensions, check the availablility\n'+
              'of data from all models for all timestamps.\n'+
              'To fix this error, set start_date and end_date as lists with the correct start and end times\nfor each model.')
        start_date = [start_date]*len(models)
        end_date = [end_date]*len(models)

    if isinstance(run_keys, str):
        run_keys = [run_keys]

    if isinstance(period, str):
        period = [period]*len(models)    
    if len(period) != len(models):
        raise ValueError(f'period must be a string or a list of the same length as models.')
    
    experiments = [f'{model}_{run_key}' for model in models for run_key in run_keys]
    missing_species = {exp : list() for exp in experiments}
    valid_experiments = {species_p: dict() for species_p in all_species}
    for exp in experiments:
        model,run_key = exp.split('_')
        filename_dict = config_data['models_info']["standard_run"][run_key]
        for species_p in all_species:
            if species_p not in filename_dict:
                missing_species[exp].append(species)
            else:
                valid_experiments[species_p][f'{model.upper()}_{filename_dict[species_p]}'] = exp

    ds_output = {}
    ds_all = {region : {exp : list() for exp in experiments} for region in regions}
    for species_p in all_species:        
        ds_tmp = read_model_output(data_dir, "flux", species_p,
                                   list(valid_experiments[species_p].keys()), 
                                   config_data, period)
        ds_in = {valid_experiments[species_p][k] : ds for k, ds in ds_tmp.items()}
        ds_in = slice_flux(ds_in, config_data, start_date, end_date, species_p,
                            country_flux_units_print = 'Tg CO2-eq yr-1')
        for region in regions:
            ds_all_region = extract_region_flux(ds_in, region)
            for exp in experiments:
                ds_all[region][exp].append(ds_all_region[exp])

    for exp in experiments :
        ds_list = []

        for region in regions:
            ds_tmp = xr.concat(align_dataset(ds_all[region][exp]), dim = 'species', combine_attrs = "no_conflicts")
            ds_list.append(ds_tmp.sum(dim='species', keep_attrs= True))

        ds_tmp =  xr.concat(ds_list, dim = 'country', combine_attrs = "no_conflicts")
        ds_tmp.attrs['species'] = species

        ds_output[exp] = ds_tmp
        
    missing = []
    for exp in experiments:
        if missing_species[exp] != []:
            missing.append(exp)
        else:
            logger.info(f' All species succesfully read for {model}!')
            
    for m in missing:
        logger.warning(f' Model {m} is missing species: {missing_species[m]}')

    logger.info(' To change the files used as the standard for each HFC/PFC, edit variable std_run in species_info.json')

    return ds_output

def load_countries_shape(
    region_bounds: tuple =None
    ) -> gpd:
    """
    Load Natural Earth vector map data and optionally filters for a specific region.

    Args:
        region_bounds (tuple, optional):
            A tuple of (min_lon, max_lon, min_lat, max_lat) to filter the map.
            Default is None, which loads the full world.

    Returns:
        gdf (GeoDataFrame): 
            A GeoDataFrame containing the country boundaries for the specified region.
    """

    # Scale of the map (1:50m)
    res = "50m"  # Can be 10m, 50m, 110m

    this_file = Path(__file__).parent.parent
    path_to_save = this_file / "data" / "ne_data"
    url = f"https://naturalearth.s3.amazonaws.com/{res}_cultural/ne_{res}_admin_0_countries.zip"
    path_to_save.mkdir(parents=True, exist_ok=True)

    shpfile = path_to_save / f"ne_{res}_admin_0_countries.shp"

    if not shpfile.is_file():
        resp = urlopen(url)
        zipfile = ZipFile(BytesIO(resp.read()))
        zipfile.extractall(path_to_save)

    gdf = gpd.read_file(shpfile)

    # Update the missing ISO_A3 values in the data
    name_to_iso_a3_mapping = {
        'Norway': 'NOR',
        'Kosovo': 'KOS',
        'France': 'FRA',
        'Indian Ocean Ter.': 'IOT',
    }

    for name, iso_a3 in name_to_iso_a3_mapping.items():
        gdf.loc[gdf['NAME'] == name, 'ISO_A3'] = iso_a3

    # If a region is specified, filter the GeoDataFrame
    if region_bounds:
        min_lon, max_lon, min_lat, max_lat = region_bounds
        gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]

    return gdf

def edit_ds(
        ds: xr.Dataset,
        model : str,
        period: str,
        file_type: str,
) -> xr.Dataset:
    
    if file_type == 'flux':
        ds = adapt_ds_flux(ds,model)

    ds = edit_ds_attributes(ds,period,file_type)

    return ds

def edit_ds_attributes(
        ds: xr.Dataset,
        period: str,
        file_type: str,
) -> xr.Dataset:
    
    # Add inversion frequency to global attributes
    if "frequency" not in ds.attrs:
        ds.attrs["frequency"] = period
    
    # Easy fix for InTEM ("units" attribute is wrongly set to "unit")
    vars_to_check = ['country_flux_total_prior', 'country_flux_total_posterior',
                        'percentile_country_flux_total_prior','percentile_country_flux_total_posterior']
    
    if file_type == 'flux':
        for var in vars_to_check:
            if 'units' not in ds[var].attrs.keys() and 'unit' in ds[var].attrs.keys():
                ds[var].attrs['units'] = ds[var].attrs['unit']

    return ds

def adapt_ds_flux(
        ds: xr.Dataset,
        model : str,
) -> xr.Dataset:
    m0 = model.split('_')[0].lower()
        
    if m0 == 'elris':
        ds['country'] = ds['country'].astype('str')
        ds = ds.set_index(countrynumber='country').rename({'countrynumber':'country'})

    elif m0 == 'intem':            
        ds = ds.rename({'countrynumber':'country'})

        if 'BEL-LUX' in ds.country and ('BEL' not in ds.country and 'LUX'  not in ds.country):
            logger.warning(f"InTEM does not estimate separate BELGIUM emissions.\n A population ratio of {config.bel_pop_r} is being used to scale InTEM's total BELGIUM+LUXEMBOURG estimate.")

            r = config.bel_pop_r

            variables_with_country = [var for var in ds.data_vars if "country" in ds[var].dims]
            numerical_vars = [var for var in variables_with_country 
                                if np.issubdtype(ds[var].dtype, np.number) and var != "country_fraction"]

            ds_bel = r * ds[numerical_vars].sel(country='BEL-LUX')
            ds_lux = (1-r) * ds[numerical_vars].sel(country='BEL-LUX')

            del ds_bel['country']
            del ds_lux['country']

            ds_bel['countryname'] = xr.DataArray(data = ['BELGIUM',] * ds_bel.time.size,
                                                dims = ['time',],
                                                coords = {'time': ds_bel.time},
                                                attrs = ds.countryname.attrs)
            
            ds_lux['countryname'] = xr.DataArray(data = ['LUXEMBOURG',] * ds_lux.time.size,
                                                dims = ['time',],
                                                coords = {'time': ds_lux.time},
                                                attrs = ds.countryname.attrs)
            
            ds_bellux = xr.concat([ds_bel, ds_lux], pd.Index(['BEL','LUX'], name='country'))
            ds = xr.merge([ds, ds_bellux])

    elif m0 == 'rhime':
        ds['country'] = [config.countrycodes_dict.get(x, x) for x in ds['country'].values]

    elif m0 == 'flexinvert':
        ds['percentile_country_flux_total_posterior'] = xr.concat([ds['country_flux_total_posterior']
                                                                - ds['country_flux_error_posterior'],
                                                                ds['country_flux_total_posterior']
                                                                + ds['country_flux_error_posterior']],
                                                                pd.Index([0,1], name = 'percentile'))
        
        ds['percentile_country_flux_total_prior'] = xr.concat([ds['country_flux_total_prior']
                                                            - ds['country_flux_error_prior'],
                                                            ds['country_flux_total_prior']
                                                            + ds['country_flux_error_prior']],
                                                            pd.Index([0,1], name = 'percentile'))
    return ds
