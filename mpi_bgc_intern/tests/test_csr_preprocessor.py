import pytest
from mpi_bgc_intern.utils import csr_preprocessor as prp
import xarray as xr
import numpy as np
import os


@pytest.fixture(scope="module")
def processed_dataset(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("data")
    path_to_prior = "mpi_bgc_intern/tests/data/Prior_vprm_flux_monthly_2006_2023_59_66km.nc"
    path_to_posterior = "mpi_bgc_intern/tests/data/Posterior_vprm_flux_monthly_2006_2023_59_66km.nc"
    output_path = tmp_path / "CSR_co2_monthly.nc"
    species = "co2"

    prp.preprocess(path_to_prior, path_to_posterior, output_path, species)

    return xr.open_dataset(output_path)


def test_output_file_is_created(processed_dataset):
    assert processed_dataset is not None


def test_variables_are_renamed_correctly(processed_dataset):
    for var in processed_dataset.data_vars:
        for new_name, old_names in prp.rename_candidates.items():
            assert var not in old_names, f"{var} has not been renamed to {new_name}"


def test_combined_variables_deleted(processed_dataset, species="co2"):
    for old_name in prp.combine_candidates:
        assert f"{old_name}_{species}" not in processed_dataset.data_vars, f"{old_name}_{species} was not deleted"


def test_combined_flux_units_correct(processed_dataset):
    for _, fluxes in prp.combine_candidates.items():
        for flux in fluxes:
            if flux in processed_dataset.data_vars:
                unit = processed_dataset[flux].attrs.get("units", None)
                assert unit == prp.flux_unit, f"Unit of {flux} is {unit}, expected {prp.flux_unit}"


def test_time_format(processed_dataset):
    dt_time = processed_dataset['time'].dtype
    assert dt_time == np.dtype("datetime64[ns]"), f"{dt_time} is the wrong format for time. Should be datetime64[ns]"