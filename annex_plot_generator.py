import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PARIS_inversion_results as func

###########################################
### GENERAL SETTINGS
###########################################
# Species to plot
monthly_species = []#['ch4']#,'n2o']

annual_species = ['hfc227ea'] #['hfc125','hfc134a','hfc143a','hfc152a','hfc23',
                  #'hfc227ea','hfc245fa','hfc32','hfc365mfc','hfc4310mee',
                  #'cf4','pfc116','pfc218','pfc318','sf6'
                  #  ]

combined_species = []#['all_hfc','all_pfc']

# Cities to plot
point_markers = {'UK': ['london','edinburgh','cardiff','belfast'],
                'SWITZERLAND': ['bern','zurich','geneva','basel','lausanne'],
                'GERMANY': ['berlin','hamburg','munich','koeln','frankfurt','essen'],
                'ITALY': ['rome','milan','naples','turin','palermo'],
                'NETHERLANDS': ['amsterdam','rotterdam','hague','utrecht','eindhoven'],
                'IRELAND': ['dublin','cork','limerick','galway','waterford'],
                'HUNGARY': ['budapest','debrecen','miskolc','szeged','pecs'],
                'NORWAY': ['oslo','bergen','sandnes','stavanger','drammen']}

# Path to results directory 
data_dir = '/project/paris/inverse_modelling/'

# Set ppt_mode to True for bigger fonts
ppt_mode = False

# Set annex_mode to True for shorter labels
annex_mode = True

###########################################

def produce_plots(regions, output_path, inventory_years):

    ### Settings for country fluxes
    models_country_fluxes = ['intem_longrun', 'intem', 'elris', 'rhime'] # NOTE: only options are basic model names w/ and w/o longrun
    scale_co2eq = True
    plot_inventory = True
    fix_y_axes = False
    add_prior = False
    add_prior_unc = False
    set_global_leg = False
    country_codes_as_titles = None
    plot_separate = [True,False,False,False] # NOTE: labels of models to plot separate might need to be adapted
    plot_combined = [False,True,True,True]
    plot_resample_and_original = False

    ### Settings for spatial maps
    models_spatial_maps = ['intem', 'elris']
    plot_area = regions[0]
    plot_site_locations = True
    plot_point_markers = point_markers[regions[0]]
    convert_flux_units = True
    set_fluxlim = 'auto'
    plot_inversion_grid_flux = True

    ### Initialization
    s_data,m_data,m_colors,annotate_coords = func.initialize_settings(ppt_mode)
    annual_res_list = list()

    ### Models for country fluxes
    models = models_country_fluxes

    period_override = None

    ### CH4 and N2O
    print('\n--- PLOTTING COUNTRY FLUXES FOR CH4/N2O ---')
    for species in monthly_species:

        # Long time window
        start_date = '2008-01-01'
        end_date   = '2023-12-01' # NOTE: there are no ELRIS CH4/N2O runs for Dec 2023

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
            ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,
                                                             scale_co2eq=scale_co2eq,species=species)[model_read]

        ### Define plotting colors
        model_colors = func.set_model_colors_2(models_std,m_colors)

        # Annual averages
        resample = 'year'

        # 1.1) Plot annual country fluxes from 2008 to 2023 from intem_longrun and combined from 3 std_run
        fig = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                     s_data,m_data,model_colors,start_date,end_date,ppt_mode,annex_mode,scale_co2eq,
                                     plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                     add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                     plot_separate=plot_separate,plot_combined=plot_combined,
                                     resample=resample,
                                     plot_resample_and_original=plot_resample_and_original,
                                     period_override=period_override)

        start_year = start_date.split('-')[0]
        end_year = '2024' #end_date.split('-')[0] #NOTE: easy fix while there are no ELRIS runs for Dec
        plot_name = f'{species}_country_flux_annual_{regions[0]}_{start_year}_{end_year}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()

        # PARIS time window
        start_date = '2018-01-01'
        end_date   = '2023-12-01' # NOTE: there are no ELRIS CH4/N2O runs for Dec 2023

        ### Re-slice the data
        for model_read in models_std:
            ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,
                                                             scale_co2eq=scale_co2eq,species=species)[model_read]

        # 1.2) Plot annual country fluxes from 2018 to 2023 from intem_longrun and combined from 3 std_run
        fig,res_dict = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                              s_data,m_data,model_colors,start_date,end_date,ppt_mode,annex_mode,scale_co2eq,
                                              plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                              add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                              plot_separate=plot_separate,plot_combined=plot_combined,
                                              resample=resample,
                                              plot_resample_and_original=plot_resample_and_original,
                                              period_override=period_override,
                                              return_res=True)

        start_year = start_date.split('-')[0]
        end_year = '2024' #end_date.split('-')[0] #NOTE: easy fix while there are no ELRIS runs for Dec
        plot_name = f'{species}_country_flux_annual_{regions[0]}_{start_year}_{end_year}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()
        
        # Store results for .csv and table
        comb = res_dict[regions[0]]['combined']
        try:
            inv = res_dict[regions[0]]['inventory']
        except:
            print('NO INVENTORY FOUND')
            inv = {'time':comb['time'],
                   'value':np.array([np.NaN,]*len(comb['time']))}
            
        tmp = {'species':[species,]*2,'source':['NIR '+inventory_years[0],'PARIS mean']}
        for it,time in enumerate(comb['time'].astype('datetime64[Y]')):
            paris_val = f"{comb['mean'][it]:.0f} \\pm {(comb['max'][it]-comb['min'][it])/2:.0f}"
            inv_val = inv['value'][inv['time'].astype('datetime64[Y]')==time]
            if len(inv_val)==1:
                tmp[str(time)] = [f'{inv_val[0]:.0f}',paris_val]                            
            else:
                 tmp[str(time)] = [None,paris_val]      
        annual_res_list.append(pd.DataFrame(tmp))

        # Monthly country fluxes
        resample = None

        # 2) Plot monthly country fluxes from 2018 to 2023 from intem_longrun and combined from 3 std_run
        fig = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                     s_data,m_data,model_colors,start_date,end_date,ppt_mode,annex_mode,scale_co2eq,
                                     plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                     add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                     plot_separate=plot_separate,plot_combined=plot_combined,
                                     resample=resample,
                                     plot_resample_and_original=plot_resample_and_original,
                                     period_override=period_override)

        start_year = start_date.split('-')[0]
        end_year = '2024' #end_date.split('-')[0] #NOTE: easy fix while there are no ELRIS runs for Dec
        plot_name = f'{species}_country_flux_monthly_{regions[0]}_{start_year}_{end_year}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()
        

    ### F-gases
    end_date   = '2024-01-01'
    resample   = None
                  

    print('\n--- PLOTTING COUNTRY FLUXES FOR ALL F-GASES ---')
    for species in annual_species:

        ds_all_flux = {}
        ds_all_flux_scaled = {}
        models_std = []

        if species == 'hfc4310mee':
            start_date = '2011-01-01' # Fix for InTEM longrun which is zero in 2010
        else:
            start_date = '2008-01-01'

        #if species == 'sf6':
            #period_override = ['monthly','yearly','yearly','yearly']
        #    period_override = ['yearly','yearly','yearly','yearly']
        #else:
        #    period_override = None
        #    model_period = None
        period_override = None
        model_period = None
        ### Read and scale fluxes
        for m,model in enumerate(models):

            m0 = model.split('_')[0]

            model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
            if 'longrun' in model: model_read = f'{model_read}_longrun'
            models_std.append(model_read)

            if period_override != None: model_period = [period_override[m]]

            # use model_read instead of model
            ds_all_flux[model_read] = func.read_flux(data_dir,species,[model_read],s_data,m_data,period_override=model_period)[model_read]
            ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,
                                                             scale_co2eq=scale_co2eq,species=species)[model_read]

        ### Define plotting colors
        model_colors = func.set_model_colors_2(models_std,m_colors)

        # 3) Plot annual country fluxes from 2008 to 2023 from intem_longrun and combined from 3 std_run
        fig,res_dict = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                              s_data,m_data,model_colors,start_date,end_date,ppt_mode,annex_mode,scale_co2eq,
                                              plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                              add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                              plot_separate=plot_separate,plot_combined=plot_combined,
                                              resample=resample,
                                              plot_resample_and_original=plot_resample_and_original,
                                              period_override=period_override,
                                              return_res=True)

        start_year = start_date.split('-')[0]
        end_year = end_date.split('-')[0]
        plot_name = f'{species}_country_flux_annual_{regions[0]}_{start_year}_{end_year}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()
        
        # Store results for .csv and table
        comb = res_dict[regions[0]]['combined']
        try:
            inv = res_dict[regions[0]]['inventory']
        except:
            print('NO INVENTORY FOUND')
            inv = {'time':comb['time'],
                   'value':np.array([np.NaN,]*len(comb['time']))}
            
        tmp = {'species':[species,]*2,'source':['NIR '+inventory_years[0],'PARIS mean']}
        for it,time in enumerate(comb['time'].astype('datetime64[Y]')):
            paris_val = f"{comb['mean'][it]:.1f} \\pm {(comb['max'][it]-comb['min'][it])/2:.1f}"
            inv_val = inv['value'][inv['time'].astype('datetime64[Y]')==time]
            if len(inv_val)==1:
                tmp[str(time)] = [f'{inv_val[0]:.1f}',paris_val]                            
            else:
                 tmp[str(time)] = [None,paris_val]      
        annual_res_list.append(pd.DataFrame(tmp))

    ### Total HFCs/PFCs (w/o HFC-4310mee)
    resample = None
    period_override = None
    start_date = ['2008-01-01','2018-01-01','2018-01-01','2018-01-01']
    end_date   = ['2024-01-01','2024-01-01','2024-01-01','2024-01-01']

    print('\n--- PLOTTING COUNTRY FLUXES FOR TOTAL HFC/PFC ---')
    for species in combined_species:

        ds_all_flux_scaled = {}

        ### Read and scale fluxes
        ds_all_flux_scaled = func.read_flux_total_fgases(data_dir,species,models,s_data,m_data,
                                                        regions,start_date,end_date,
                                                        period_override=period_override)

        ### Define plotting colors
        model_colors = func.set_model_colors_2(models,m_colors)

        # 4) Plot annual country fluxes from 2008 to 2023 from intem_longrun and combined from 3 std_run
        fig,res_dict = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                              s_data,m_data,model_colors,start_date,end_date,ppt_mode,annex_mode,scale_co2eq,
                                              plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                              add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                              plot_separate=plot_separate,plot_combined=plot_combined,
                                              resample=resample,
                                              plot_resample_and_original=plot_resample_and_original,
                                              period_override=period_override,
                                              return_res=True)
        start_year = start_date[0].split('-')[0]
        end_year = end_date[0].split('-')[0]
        plot_name = f'{species}_country_flux_annual_{regions[0]}_{start_year}_{end_year}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()
                              
        # Store results for .csv and table
        comb = res_dict[regions[0]]['combined']
        try:
            inv = res_dict[regions[0]]['inventory']
        except:
            print('NO INVENTORY FOUND')
            inv = {'time':comb['time'],
                   'value':np.array([np.NaN,]*len(comb['time']))}
            
        tmp = {'species':[species,]*2,'source':['NIR '+inventory_years[0],'PARIS mean']}
        for it,time in enumerate(comb['time'].astype('datetime64[Y]')):
            paris_val = f"{comb['mean'][it]:.1f} \\pm {(comb['max'][it]-comb['min'][it])/2:.1f}"
            inv_val = inv['value'][inv['time'].astype('datetime64[Y]')==time]
            if len(inv_val)==1:
                tmp[str(time)] = [f'{inv_val[0]:.1f}',paris_val]                            
            else:
                 tmp[str(time)] = [None,paris_val]      
        annual_res_list.append(pd.DataFrame(tmp))

    ### Models for spatial maps
    models = models_spatial_maps

    # Settings for average posterior
    cmap = 'viridis'
    c_border = 'floralwhite'
    var = 'flux_total_posterior'
    plot_combined = True
    period_override = None

    start_date  = '2018-01-01'
    all_species = monthly_species + annual_species

    # All species
    print('\n--- PLOTTING MEAN POSTERIOR MAP FOR ALL SPECIES ---')
    for species in all_species:

        ds_all_flux = {}
        ds_all_flux_scaled = {}
        models_std = []

        if species in monthly_species:
            end_date = '2024-01-01'
            chop_by = ['2018-01-01'] # NOTE: this setting can deal with no ELRIS CH4/N2O runs in Dec 2023
            dt = None
        elif species == "hfc4310mee":
            end_date = '2023-01-01'
            chop_by = 'year'
            dt = 5
        else:
            end_date = '2024-01-01'
            chop_by = 'year'
            dt = 6

        ### Read and scale fluxes
        for m,model in enumerate(models):

            m0 = model.split('_')[0]

            model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
            models_std.append(model_read)

            # use model_read instead of model
            ds_all_flux[model_read] = func.read_flux(data_dir,species,[model_read],s_data,m_data,period_override=period_override)[model_read]
            ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,
                                                             convert_flux_units=convert_flux_units,species=species)[model_read]

        # 5) Plot spatial map of the posterior fluxes averaged between 2018 and 2023 (combined from 3 std_run)
        fig = func.plot_spatial_flux_per_timestamp(ds_all_flux_scaled,species,plot_area,end_date,s_data,m_data,
                                                    cmap=cmap,c_border=c_border,var=var,
                                                    plot_combined=plot_combined, annex_mode=annex_mode,
                                                    chop_by=chop_by,dt=dt,period_override=period_override,
                                                    plot_site_locations=plot_site_locations,
                                                    plot_point_markers=plot_point_markers,
                                                    set_fluxlim=set_fluxlim,
                                                    plot_inversion_grid_flux=plot_inversion_grid_flux)

        start_year = start_date.split('-')[0]
        end_year = end_date.split('-')[0]
        plot_name = f'{species}_posterior_map_{regions[0]}_{start_year}_{end_year}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()

    # Settings for seasonal difference to the mean
    cmap = 'coolwarm'
    c_border = 'dimgrey'
    chop_by = 'season'
    dt = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
    var = 'posterior_mean_diff'
    plot_combined = True

    # CH4 and N2O
    start_date = '2018-01-01'
    end_date   = '2024-01-01'

    print('\n--- PLOTTING SEASONAL POSTERIOR MAP FOR CH4/N2O ---')
    for species in monthly_species:

        ds_all_flux = {}
        ds_all_flux_scaled = {}
        models_std = []

        ### Read and scale fluxes
        for m,model in enumerate(models):

            m0 = model.split('_')[0]

            model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
            models_std.append(model_read)

            # use model_read instead of model
            ds_all_flux[model_read] = func.read_flux(data_dir,species,[model_read],s_data,m_data,period_override=period_override)[model_read]
            ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,
                                                             convert_flux_units=convert_flux_units,species=species)[model_read]

        # 6) Plot spatial maps of the seasonal posterior fluxes (averaged between 2018 and 2023) subtracted by the mean (combined from 3 std_run)
        fig = func.plot_spatial_flux_per_timestamp(ds_all_flux_scaled,species,plot_area,end_date,s_data,m_data,
                                                    cmap=cmap,c_border=c_border,var=var,
                                                    plot_combined=plot_combined,annex_mode=annex_mode,
                                                    chop_by=chop_by,dt=dt,period_override=period_override,
                                                    plot_site_locations=plot_site_locations,
                                                    plot_point_markers=plot_point_markers,
                                                    set_fluxlim = set_fluxlim,
                                                    plot_inversion_grid_flux=plot_inversion_grid_flux)

        start_year = start_date.split('-')[0]
        end_year = end_date.split('-')[0]
        plot_name = f'{species}_seasonal_map_{regions[0]}_{start_year}_{end_year}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()

    print('\n--- ALL PLOTS GENERATED SUCCESSFULLY! ---')
                              
                              
    print('\n\n\n--- GENERATING TABLES ---')
    annual_res = pd.concat(annual_res_list).reset_index(drop=True).fillna(value=' ')
    
    print('\n\nTABLE HFC\n\n')
    hfc_res = annual_res[annual_res.species.apply(lambda x : x[:3].lower()=='hfc')].copy()
    hfc_res['species'] = hfc_res.species.apply(lambda x : x.replace('hfc','HFC-'))
    make_table(hfc_res,f'{output_path}/hfc_res_{regions[0]}.tex')
    hfc_res.to_csv(f'{output_path}/hfc_res_{regions[0]}.csv',index=False)
    
    print('\n\nTABLE PFC\n\n')
    pfc_res = annual_res[annual_res.species.apply(lambda x : x[:3].lower()=='pfc' or x.lower()=='cf4')].copy()
    pfc_res['species'] = pfc_res.species.apply(lambda x : x.replace('pfc','PFC-'))
    make_table(pfc_res,f'{output_path}/pfc_res_{regions[0]}.tex')
    pfc_res.to_csv(f'{output_path}/pfc_res_{regions[0]}.csv',index=False)
    
    print('\n\nTABLE main gases\n\n')
    
    main_gases_res = annual_res[annual_res.species.isin(['ch4','n2o','sf6','all_pfc','all_hfc'])].copy()
    main_gases_res['species'] = main_gases_res.species.apply(lambda x : x.upper().replace('ALL_','Total '))
    make_table(main_gases_res,f'{output_path}/main_gases_res_{regions[0]}.tex')
    main_gases_res.to_csv(f'{output_path}/main_gases_res_{regions[0]}.csv',index=False)
    
                              
    print('\n--- TABLES GENERATED SUCCESSFULLY! ---')
    return annual_res
    

def make_table(df,output_path,
               descriptive_cols=['species','source'],
               hline_place={'source':'PARIS mean'}
              ):
    if 'hfc' in output_path:
        species = 'HFCs'
    elif 'pfc' in output_path:
        species = 'PFCs'
    if 'main_gases' in output_path:
        species = 'the main greenhouse gases of focus'
    # Set latex Table env and number of cols
    tmp = output_path.split('/')[-1].split('.')[0]
    label = '\n \\label{'+tmp+'}'
    tmp = "Emissions estimation for "+species+" in $\\rm{TgCO}_{2}\\rm{-eq} \\cdot \\rm{yr}^{-1}$ according to the National Inventory Report (NIR) 2024 and the inversions done in the PARIS project. For the PARIS estimation, the mean of the 3 inversion models is displayed, along with a range of uncertainty estimated via the half distance between the maximum and minimum uncertainties of the different models."  
    caption = '\n \\caption{'+tmp+'}'
    begin = '\\begin{table}'+label+caption+'\n \\begin{center}\n  \\begin{tabular}{ '+len(descriptive_cols)*'l '+(len(df.columns)-len(descriptive_cols))*'r '+'}'
    
    # Set first line with columns title
    header = '     '+len(descriptive_cols)*' & '
    for y in df.columns[len(descriptive_cols):]:
        header += y
        if y!=df.columns[-1]: 
            header+=' & '
    
    table = begin + '\n' + header + ' \\\\ \hline' + '\n'
    
    # Iterate over lines of dataframe
    prev_species = ''
    for idRow,row in df.iterrows():
        # Indentation
        l = '    '
        
        # Test if value for first column needed
        if row[descriptive_cols[0]]==prev_species:
            l += ' & '
        else:
            l += row[descriptive_cols[0]]+' & '
        prev_species = row[descriptive_cols[0]]
        
        # Add values for other descriptive columns
        for col in descriptive_cols[1:]:
            l += row[col]+' & '
        
        # Add yearly values
        for y in df.columns[len(descriptive_cols):]:
            l += '$ '+row[y]+' $'
            if y!=df.columns[-1]: 
                l+=' & '
        
        # End line
        l += ' \\\\ '
        
        # Add hline if needed
        for key in hline_place.keys():
            if row[key]==hline_place[key]:
                l += ' \hline '
        
        # Add line to table
        table+= l+'\n '
    
    # Close latex env
    end = str('  \\end{tabular}\n \\end{center}\n\\end{table}')
    
    table += end
    
    with open(output_path, "w") as text_file:
        text_file.write(table)
