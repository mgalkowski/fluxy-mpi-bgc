from pathlib import Path
import fluxy
from fluxy.config import initialize_settings
from fluxy.io import read_flux, read_flux_total_fgases
from fluxy.operators.select import slice_flux


data_dir = Path(fluxy.__path__[0]).parent / "data" / "tests"



def deactivated_test_read_data():
    # This test fails sometimes when runned wil all the other tests
    # Because of a xarray cache problem
    # Probably once we move the code that read the data in the plot test script
    # we can reactivate this and it will work
    
    config_data, m_colors, annotate_coords = initialize_settings()

    species = "hfc134a"  # options for individual species, or 'all_hfc' or 'all_pfc'
    models = ["intem_name_edgar", "elris_name_edgar", "rhime_name_edgar"]
    scale_co2eq = False
    period_override = None  # use to override standard inversion periods, must be a list the same length as models, e.g. ['monthly','yearly']
    start_date = "2018-01-01"  # inclusive. Option to set as list of dates, e.g. ['2018-01-01','2019-01-01'] which is required for total fgases if one model is missing obs for a year
    end_date = "2024-01-01"  # not inclusive. Option to set as list of dates, e.g. ['2023-01-01','2022-01-01'] which is required for total fgases if one model is missing obs for a year

    ds_all_flux_scaled = {}

    ds_all_flux = read_flux(
        data_dir,
        species,
        models,
        config_data["species_info"],
        config_data["models_info"],
        period_override=period_override,
    )

    for m in models:
        ds_all_flux_scaled[m] = slice_flux(
            {m: ds_all_flux[m]},
            start_date,
            end_date,
            config_data,
            scale_units=True,
            scale_co2eq=scale_co2eq,
            convert_flux_units=False,
            specie=species,
        )[m]
