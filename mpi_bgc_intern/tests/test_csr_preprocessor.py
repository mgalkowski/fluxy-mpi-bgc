import pytest
from mpi_bgc_intern.utils import csr_preprocessor as prp
import xarray as xr
import os
import numpy as np


def test_output_file_is_created(tmp_path):
    path_to_prior = "mpi_bgc_intern/tests/data/Prior_vprm_flux_monthly_2006_2023_59_66km.nc"
    path_to_posterior = "mpi_bgc_intern/tests/data/Posterior_vprm_flux_monthly_2006_2023_59_66km.nc"
    path_to_output = tmp_path / "output.nc"
    species = "co2"

    prp.preprocess(path_to_prior, path_to_posterior, path_to_output, species)

    assert path_to_output.exists(), "Output file was not created."


def test_variables_are_renamed_correctly():
    ds = xr.open_dataset("mpi_bgc_intern/tests/data/Output/CSR/co2/CSR_co2_monthly.nc")

    for var in ds.data_vars:
        for new_name, old_names in prp.rename_candidates.items():
            assert var not in old_names, f"{var} has not been renamed to {new_name}"

def test_combined_variables_deleted(species="co2"):
    ds = xr.open_dataset("mpi_bgc_intern/tests/data/Output/CSR/co2/CSR_co2_monthly.nc")

    for old_name in prp.combine_candidates:
        assert f"{old_name}_{species}" not in ds.data_vars, f"{old_name}_{species} was not deleted"

def test_combined_flux_units_correct():
    ds = xr.open_dataset("mpi_bgc_intern/tests/data/Output/CSR/co2/CSR_co2_monthly.nc")

    for _, fluxes in prp.combine_candidates.items():
        for flux in fluxes:
            if flux in ds.data_vars:
                unit = ds[flux].attrs.get("units", None)
                assert unit == prp.flux_unit, f"Unit of {flux} is {unit}, expected {prp.flux_unit}"



def test_time_format():
    ds = xr.open_dataset("mpi_bgc_intern/tests/data/Output/CSR/co2/CSR_co2_monthly.nc")
    dt_time = ds['time'].dtype
    assert dt_time == np.dtype("datetime64[ns]"), f"{dt_time} is the wrong format for time. Should be {np.dtype("datetime64[ns]")}"