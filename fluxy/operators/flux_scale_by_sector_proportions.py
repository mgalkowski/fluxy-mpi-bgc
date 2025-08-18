import logging
import os
import numpy as np
import xarray as xr
from pathlib import Path
from fluxy.operators.convert import get_units_conversion_factor
from fluxy.operators.flux_align_dataset import align_lat_lon

logger = logging.getLogger(__name__)

def scale_by_sector_proportions(data_dir:str,
                                ds: xr.Dataset,
                                species: str,
                                sectors: list[str],
                                country_flux_units_print: str,
                                config_data = dict[str, dict],
                                sector_file: str = 'EUROPE_EDGAR',
                                create_region_sector_totals: bool = True
) -> xr.Dataset:
    """
    Produces sector level fluxes by scaling prior and posterior fluxes by the 
    proportional contribution of each sector to the total flux in each grid cell
    (at transport model resolution). Fluxes are then summed over region/country areas
    to produce region/country sector totals.    
    
    Args:
        data_dir (str):
            Path to top data directory.
        ds (xarray dataset):
            xarray dataset with model data.
        species (str):
            Gas species, e.g. 'ch4'.
        sectors (list of str):
            Emissions sectors to include, options for 'agriculture', 'waste', 
            'energy' and 'industry'.
        country_flux_units_print (str):
            Units for country flux, e.g. 'Gg yr-1'
        config_data (dict):
            config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        sector_file (str):
            Start of sector file name, e.g. 'EUROPE_EDGAR'
        create_region_sector_totals (bool):
            If True, sums spatial fluxes over country_fraction masks 
            to create country/region totals.
    
    Returns:
        ds (xarray dataset):
            Input ds, with added flux_sector_prior and flux_sector_posterior variables.
            If regions is not None, also contains region/country sector total variables.
    """
     
    sector_prop_path = os.path.join(data_dir,'sector_flux',
                                       f'{sector_file}_{species}_yearly_flux_sectors.nc')
    
    with xr.open_dataset(sector_prop_path) as f:
        # timely align sector dataset on main dataset
        freq_ds = ds.attrs["frequency"]
        if xr.infer_freq(f.time) != 'YS-JAN' or freq_ds not in ["monthly", "yearly"]:
            raise ValueError("This part of the code as not be tested for a sector file with frequency not equal to 'YS-JAN' or dataset frequency not one of 'monthly'/'yearly'. In the current implementation, the sector file frequency is suppose to be bugger or equal to the dataset frequency and start before or at the same time as the dataset.")
        ds_sectors = f.sel(time=ds["time"].values, method="ffill")
        ds_sectors["time"] = ds["time"]

        # spatially align sector dataset on main dataset
        ds_sectors = ds_sectors.rename({"lat":"latitude",
                                        "lon":"longitude"})
        _, ds_sectors = align_lat_lon([ds,ds_sectors],"latitude")
        _, ds_sectors = align_lat_lon([ds,ds_sectors],"longitude")

    # Convert prior and posterior flux for each sector
    for s in sectors:
        scaling_factor = ds_sectors[f"flux_{s}"]/ds_sectors['flux_total']
        scaling_factor = scaling_factor.where(ds_sectors['flux_total']!=0,0)
        
        ds[f'flux_{s}_prior'] = ds['flux_total_prior']*scaling_factor
        ds[f'flux_{s}_prior'].attrs = {'units':ds['flux_total_prior'].attrs["units"],
                                            '_FillValue':np.nan,
                                            'long_name':f'prior {s} {species} flux, created by scaling total flux by sector proportions'}
        
        ds[f'flux_{s}_posterior'] = ds['flux_total_posterior']*scaling_factor
        ds[f'flux_{s}_posterior'].attrs = {'units':ds['flux_total_posterior'].attrs["units"],
                                            '_FillValue':np.nan,
                                            'long_name':f'posterior total {species} {s} flux, created by scaling total flux by sector proportions'}

    # Convert prior and posterior country flux for each sector if needed
    if create_region_sector_totals:
        # derive factor for unit conversion
        molar_mass = config_data["species_info"][species]["molar_mass"]
        units_factor = get_units_conversion_factor(ds[f'flux_{s}_prior'].attrs["units"].replace(" m-2 ", " "), 
                                                country_flux_units_print, 
                                                molar_mass)

        # check for crountry_fraction
        if 'country_fraction' not in ds:
            raise ValueError(
                f"country_fraction variable not present in dataset for {ds['inversion_system']} "
                "Cannot add sector country totals."
                )

        # check for cell_area and create if needed
        if 'cell_area' not in ds:
            parent_dir = Path(__file__).parent.parent.parent
            configs_dir = parent_dir / "configs"
            if 'domain' in ds.attrs:
                domain = ds.attrs["domain"]
            else:
                domain = sector_file.split('_')[0]
                logger.warning(f'No domain info in dataset attributes, so reading domain from sector_filename: {domain}')
            with xr.open_dataset(os.path.join(configs_dir,f"{domain}_cell_area.nc")) as f:
                cell_area = f.cell_area

            _, cell_area = align_lat_lon([ds,cell_area],"latitude")
            _, cell_area = align_lat_lon([ds,cell_area],"longitude")
            ds["cell_area"] = cell_area

        logger.warning("cell_area should be in meter square and flux in something per m-2. No check is made on this for the moment.")

        # calculate sector country flux (prior/posterior) from sector flux (prior/posterior), cell_area, country_fraction and unit_factor
        for s in sectors:
            scaling_factor = ds_sectors[f"flux_{s}"]/ds_sectors['flux_total']
            scaling_factor = scaling_factor.where(ds_sectors['flux_total']!=0,0)
            
            ds[f'flux_{s}_prior_country'] = (ds[f'flux_{s}_prior']*ds['country_fraction']*ds["cell_area"]).sum(dim=['latitude', 'longitude'])*units_factor
            ds[f'flux_{s}_prior_country'].attrs = {'units':country_flux_units_print,
                                                '_FillValue':np.nan,
                                                'long_name':f'country {s} {species} prior flux, created by scaling total flux by sector proportions'}
            
            ds[f'flux_{s}_posterior_country'] = (ds[f'flux_{s}_posterior']*ds['country_fraction']*ds["cell_area"]).sum(dim=['latitude', 'longitude'])*units_factor
            ds[f'flux_{s}_posterior_country'].attrs = {'units':country_flux_units_print,
                                                '_FillValue':np.nan,
                                                'long_name':f'country {s} {species} posterior flux, created by scaling total flux by sector proportions'}
    return ds