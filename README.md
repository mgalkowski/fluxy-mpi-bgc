# PARIS_intercomparison

This repository contains functions to compare inverse models used during the PARIS project, and a notebook to allow for easy use of these functions.

To run the notebook and plot model results and comparisons:

1. Edit the first notebook cell, which specifies the filepaths for each model, and the model names and colours used for plotting. In these variables, you can specify multiple experiment runs, as shown in the example notebook.

2. To select the models you want to plot, edit the variables specifying species, model names, dates etc. under each numbered notebook heading. This cell will then read in the data, select values between your chosen dates and filter for baseline timestamps (if you choose this option).

3. Run the subsequent cells under each numbered heading to produce various plots, as detailed in the example notebook. The mole fraction plot functions allow you to plot any number of variables, specified by the `include` and `diff_include` variables.
