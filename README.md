# PARIS inverse modelling intercomparison tool

This repository contains functions to compare inverse models used during the PARIS project, and a notebook to allow for easy use of these functions.

Follow the steps below to run the notebook and plot model results.

## Clone repo and install package in ICOS Carbon Portal

Clone the repository, install fluxy, and check if fluxy was correctly installed in your home directory:
```
git clone https://github.com/ACRG-Bristol/PARIS_intercomparison_restructure.git
cd PARIS_intercomparison_restructure
pip install -e .
pip show fluxy
```
Restart the kernel (otherwise package fluxy won't be found).

## Upload input files
### 1. Flux and concentration netCDF files with model results

Data format must be in agreement with the PARIS-AVENGERS-EYECLIMA template.

Filenames should follow the following format:  
`<inversionModel>_<transportModel>_<domain>_<prior>_<optional_tags>_<species>_<inversionFrequency>(_concentration).nc`

Note that the part within parenthesis refers to the concentration file only.
`<inversionFrequency>` should be equal to "yearly" or "monthly".

e.g.:  
InTEM_NAME_EUROPE_EDGAR_4sites_hfc134a_yearly.nc  
InTEM_NAME_EUROPE_EDGAR_4sites_hfc134a_yearly_concentration.nc

The following folder structure is expected:
`/path/to/data/<inversionModel>/<species>/`

In the notebook, you can specify the pair of output files that you want to plot by providing the following sequence of name tags:
`<inversionModel>_<transportModel>_<prior>_<optional_tags>`

e.g.: "InTEM_NAME_EDGAR_4sites"

Note that `<domain>` is not specified in the name of the model run because it is defined in models_info.json (see below).
`<species>` and `<inversionFrequency>` are not specified in the model run name because they are defined in designated variables in the notebook.

### 2. Models information (file "models_info.json")

Example file located in folder configs/.

| Variables     | Type                      | Description  |
|:--------------|:--------------------------|:-------------|
| domain        | str                       | Domain name tag used in the filename (e.g. "EUROPE"). |
| filename_tags | dict[str,str] (optional)  | Dictionary of keys that point to long filename strings. <br> Used to reduce the sequence of name tags that constitute the model run name. <br> e.g. if {"std" : "4sites_baseline_optimized"}, model run name "InTEM_NAME_EDGAR_std" points to "InTEM_NAME_EDGAR_4sites_baseline_optimized". <br> Note that `<model>` can be used as a generic tag which will be replaced by the inversion model name in lower case. |
| model_labels  | dict[str, str] (optional) | Dictionary with model run names and respective labels to use in the plot. <br> If not defined, the label is created automatically from the model run names. |
| species_name  | dict[str,dict] (optional) | Species name that should replace `<species>` in the filename. <br> By default, `<species>` is assumed equal to the value specified in the notebook (e.g. "hfc134a"). <br> Use this dictionary to specify model specific species name. Dictionary keys should correspond to `<inversionModel>`. |
| standard_run  | dict[str,dict]            | Name tags (`<transportModel>_<prior>_<optional_tags>`) that identify the standard run for all models and each gas. <br> These runs are considered when summing country fluxes from all HFCs or PFCs (e.g. option species="all_hfc"). <br> To use the name tags specified under "default", specify only the `<inversionModel>` name in the notebook (e.g. models=["InTEM","RHIME"]). <br> Define other dictionary keys (e.g. "longrun") to specify a different set of model runs. You can point to these runs by specifying `<inversionModel>_<key_name>` (e.g. models=["InTEM_longrun"]). Missing species will be taken from the "default" dictionary. |

### 3. Species information (file "species_info.json")

Example file located in folder configs/.

It contains a dictionary of species (or group of species) pointing to various properties/print settings.

| Keys          | Type                   | Description                                                                                                         |
|:--------------|:-----------------------|:--------------------------------------------------------------------------------------------------------------------|
| species_print | str                    | Species name used in the plot axis (LaTeX format).                                                                  |
| gwp           | float (optional)       | Global Warming Potential. <br> Used to convert country fluxes to mass of CO2 equivalent.                            |
| molar_mass    | float (optional)       | Species molar mass (g mol-1). <br> Used to apply mol<->g conversion to fluxes.                                      |
| list_species  | list of str (optional) | List of species which define a given group of species. <br> Used to plot sum of country fluxes over various species.|

### 4. Sites information (file "site_info.json")

Example file located in folder configs/.

It contains a dictionary of stations (station designation code) pointing to various station specifications.

Keys for each station:

[to be completed]

### 5. netCDF file with country fluxes from bottom-up inventory (optional)

[to be completed]

### 6. netCDF file with baseline timestamps (optional)

[to be completed]

## Run the notebook

The notebook that allows you plot the different variables of interest is located in:
`scripts/PARIS_inversion_results.ipynb`

1. Edit the first notebook cell, which specifies the filepath. You can specify multiple experiment runs, as shown in the example notebook.

2. To select the models you want to plot, edit the variables specifying species, model names, dates etc. under each numbered notebook heading. This cell will then read in the data, select values between your chosen dates and filter for baseline timestamps (if you choose this option).

3. Run the subsequent cells under each numbered heading to produce various plots, as detailed in the example notebook. Plotting options are described in front of each variable.
