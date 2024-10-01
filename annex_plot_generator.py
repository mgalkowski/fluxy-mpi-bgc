import os
import PARIS_inversion_results as func

### Define region of interest
regions = ['UK'] # Focus countries: UK, SWITZERLAND, GERMANY, ITALY, NETHERLANDS, IRELAND, HUNGARY, NORWAY

### Set output path
output_path = '/project/paris/users/Daniela/plots_draft_annex/'

###########################################
### SETTINGS
###########################################
### Global settings
# Species to plot
monthly_species = ['ch4','n2o']

annual_species = ['hfc125','hfc134a','hfc143a','hfc152a','hfc23',
                  'hfc227ea','hfc245fa','hfc32','hfc365mfc','hfc4310mee',
                  'cf4','pfc116','pfc218','pfc318','sf6']

combined_species = ['all_hfc','all_pfc']

# Cities to plot
point_markers = {'UK': ['london'],
                 'SWITZERLAND': ['bern'],
                 'GERMANY': ['berlin'],
                 'ITALY': ['rome'],
                 'NETHERLANDS': ['amsterdam'],
                 'IRELAND': ['dublin'],
                 'HUNGARY': ['budapest'],
                 'NORWAY': ['oslo']} # TODO: write cities to mark (highly populated cities)

# Path to results directory 
data_dir = '/project/paris/inverse_modelling/'

# Set ppt_mode to True for bigger fonts
ppt_mode = False

### Settings for country fluxes
models_country_fluxes = ['intem_longrun', 'intem', 'elris', 'rhime'] # NOTE: only options are basic model names w/ and w/o longrun
plot_inventory = True
inventory_years = None
fix_y_axes = False
add_prior = False
add_prior_unc = False
set_global_leg = False
country_codes_as_titles = None
plot_separate = [True,False,False,False] # NOTE: labels of models to plot separate might need to be adapted
plot_combined = [False,True,True,True]
plot_resample_and_original = False
period_override = None
# TODO: make necessary changes to plot in CO2-eq (use GWP in species_info?)

### Settings for spatial maps
models_spatial_maps = ['intem', 'elris', 'rhime']
plot_area = regions[0]        # TODO: define options for all focus countries
plot_site_locations = True
plot_point_markers = point_markers[regions[0]]
# TODO: implement combined option to average over all models [plot_combined = True]
# TODO: add all cities in point_markers and their coordinates to point_source_dict

###########################################

### Initialization
s_data,m_data,m_colors,annotate_coords = func.initialize_settings(ppt_mode)

### Models for country fluxes
models = models_country_fluxes

### CH4 and N2O
for species in monthly_species:

    # Long time window
    start_date = '2008-01-01'
    end_date   = '2024-01-01'

    ds_all_flux = {}
    ds_all_flux_scaled = {}
    models_std = []

    ### Read and scale fluxes
    for m,model in enumerate(models):

        m0 = model.split('_')[0]

        model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
        if 'longrun' in model: model_read = f'{model_read}_longrun'
        models_std.append(model_read)

        # use model_read instead of model
        ds_all_flux[model_read] = func.read_flux(data_dir,species,[model_read],s_data,m_data,period_override=period_override)[model_read]
        ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,species=species)[model_read]

    ### Define plotting colors
    model_colors = func.set_model_colors_2(models_std,m_colors)

    # Annual averages
    resample = 'year'

    # 1.1) Plot annual country fluxes from 2008 to 2023 from intem_longrun and combined from 3 std_run
    fig = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                 s_data,m_data,model_colors,start_date,end_date,ppt_mode,
                                 plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                 add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                 plot_separate=plot_separate,plot_combined=plot_combined,
                                 resample=resample,
                                 plot_resample_and_original=plot_resample_and_original,
                                 period_override=period_override)

    start_year = start_date.split('-')[0]
    end_year = end_date.split('-')[0]
    plot_name = f'{species}_country_flux_annual_{regions[0]}_{start_year}_{end_year}.png'
    full_path = os.path.join(output_path, plot_name)
    fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)

    # PARIS time window
    start_date = '2018-01-01'
    end_date   = '2024-01-01'

    ### Re-slice the data
    for model_read in models_std:
        ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,species=species)[model_read]

    # 1.2) Plot annual country fluxes from 2018 to 2023 from intem_longrun and combined from 3 std_run
    fig = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                 s_data,m_data,model_colors,start_date,end_date,ppt_mode,
                                 plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                 add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                 plot_separate=plot_separate,plot_combined=plot_combined,
                                 resample=resample,
                                 plot_resample_and_original=plot_resample_and_original,
                                 period_override=period_override)

    start_year = start_date.split('-')[0]
    end_year = end_date.split('-')[0]
    plot_name = f'{species}_country_flux_annual_{regions[0]}_{start_year}_{end_year}.png'
    full_path = os.path.join(output_path, plot_name)
    fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)

    # Monthly country fluxes
    resample = None

    # 2) Plot monthly country fluxes from 2018 to 2023 from intem_longrun and combined from 3 std_run
    fig = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                 s_data,m_data,model_colors,start_date,end_date,ppt_mode,
                                 plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                 add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                 plot_separate=plot_separate,plot_combined=plot_combined,
                                 resample=resample,
                                 plot_resample_and_original=plot_resample_and_original,
                                 period_override=period_override)

    start_year = start_date.split('-')[0]
    end_year = end_date.split('-')[0]
    plot_name = f'{species}_country_flux_monthly_{regions[0]}_{start_year}_{end_year}.png'
    full_path = os.path.join(output_path, plot_name)
    fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)

### F-gases
start_date = '2008-01-01'
end_date   = '2024-01-01'
resample   = None

for species in annual_species:

    ds_all_flux = {}
    ds_all_flux_scaled = {}
    models_std = []

    ### Read and scale fluxes
    for m,model in enumerate(models):

        m0 = model.split('_')[0]

        model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
        if 'longrun' in model: model_read = f'{model_read}_longrun'
        models_std.append(model_read)

        # use model_read instead of model
        ds_all_flux[model_read] = func.read_flux(data_dir,species,[model_read],s_data,m_data,period_override=period_override)[model_read]
        ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,species=species)[model_read]

    ### Define plotting colors
    model_colors = func.set_model_colors_2(models_std,m_colors)

    # 3) Plot annual country fluxes from 2008 to 2023 from intem_longrun and combined from 3 std_run
    fig = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                 s_data,m_data,model_colors,start_date,end_date,ppt_mode,
                                 plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                 add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                 plot_separate=plot_separate,plot_combined=plot_combined,
                                 resample=resample,
                                 plot_resample_and_original=plot_resample_and_original,
                                 period_override=period_override)

    start_year = start_date.split('-')[0]
    end_year = end_date.split('-')[0]
    plot_name = f'{species}_country_flux_annual_{regions[0]}_{start_year}_{end_year}.png'
    full_path = os.path.join(output_path, plot_name)
    fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)

### Total HFCs/PFCs
start_date = ['2008-01-01','2018-01-01','2018-01-01','2018-01-01']
end_date   = ['2024-01-01','2024-01-01','2024-01-01','2024-01-01']

for species in combined_species:

    ds_all_flux_scaled = {}

    ### Read and scale fluxes
    ds_all_flux_scaled = func.read_flux_total_fgases(data_dir,species,models,s_data,m_data,
                                                    regions,start_date,end_date,
                                                    period_override=period_override)

    ### Define plotting colors
    model_colors = func.set_model_colors_2(models,m_colors)

    # 4) Plot annual country fluxes from 2008 to 2023 from intem_longrun and combined from 3 std_run
    fig = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                s_data,m_data,model_colors,start_date,end_date,ppt_mode,
                                plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                plot_separate=plot_separate,plot_combined=plot_combined,
                                resample=resample,
                                plot_resample_and_original=plot_resample_and_original,
                                period_override=period_override)

    start_year = start_date[0].split('-')[0]
    end_year = end_date[0].split('-')[0]
    plot_name = f'{species}_country_flux_annual_{regions[0]}_{start_year}_{end_year}.png'
    full_path = os.path.join(output_path, plot_name)
    fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
