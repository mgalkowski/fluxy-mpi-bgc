from fluxy.io import read_config_files
import os
import numpy as np
import matplotlib.pyplot as plt
from json import load
from pathlib import Path
import logging


model_q_indices = {'intem':[0,1],
                   'rhime':[0,1],
                   'elris':[0,1],
                   'flexinvert':[0,1]}

point_source_dict = {
                    'paris':[2.3404,48.8600],
                    'nw_england':[-2.7969,53.7748],
                    'london':[-0.1278, 51.5074],
                    'edinburgh':[-3.1883, 55.9533],
                    'cardiff':[-3.1791, 51.4816],
                    'belfast':[-5.9301, 54.5973],
                    'zurich': [8.5417, 47.3769],
                    'geneva': [6.1432, 46.2044],
                    'basel': [7.5886, 47.5596],
                    'lausanne': [6.6323, 46.5197],
                    'bern': [7.4474, 46.9481],
                    'berlin': [13.4050, 52.5200],
                    'hamburg': [9.9937, 53.5511],
                    'munich': [11.5820, 48.1351],
                    'koeln': [6.9603, 50.9375],
                    'frankfurt': [8.6821, 50.1109],
                    'essen': [7.0115, 51.4556],
                    'rome': [12.4964, 41.9028],
                    'milan': [9.1900, 45.4642],
                    'naples': [14.2681, 40.8518],
                    'turin': [7.6869, 45.0703],
                    'palermo': [13.3615, 38.1157],
                    'amsterdam': [4.9041, 52.3676],
                    'rotterdam': [4.4777, 51.9244],
                    'hague': [4.3007, 52.0705],
                    'utrecht': [5.1214, 52.0907],
                    'eindhoven': [5.4797, 51.4416],
                    'dublin': [-6.2603, 53.3498],
                    'cork': [-8.4727, 51.8985],
                    'limerick': [-8.6305, 52.6638],
                    'galway': [-9.0579, 53.2707],
                    'waterford': [-7.1119, 52.2593],
                    'budapest': [19.0402, 47.4979],
                    'debrecen': [21.6273, 47.5316],
                    'szeged': [20.1488, 46.2530],
                    'miskolc': [20.7852, 48.0993],
                    'pecs': [18.2324, 46.0784],
                    'oslo': [10.7522, 59.9139],
                    'bergen': [5.3241, 60.3929],
                    'sandnes': [58.8514, 58.8500],
                    'stavanger': [5.7382, 58.9690],
                    'drammen': [10.2045, 59.7438],
                    'brussels':[4.3525, 50.8466],
                    'antwerp':[4.4002, 51.2177],
                    'ghent':[3.7252, 51.0536],
                    'charleroi':[4.4444, 50.4113],
                    'liege':[5.5674, 50.6337],
                    'luxembourg':[6.1319, 49.6116]
                    }

countrycodes_dict = {'IRELAND':'IRL',
                     'UK':'GBR',
                     'FRANCE':'FRA',
                     'NETHERLANDS':'NLD',
                     'GERMANY':'DEU',
                     'DENMARK':'DNK',
                     'SWITZERLAND':'CHE',
                     'AUSTRIA':'AUT',
                     'ITALY':'ITA',
                     'BELGIUM': 'BEL',
                     'LUXEMBOURG': 'LUX',
                     'HUNGARY':'HUN',
                     'SWEDEN':'SWE',
                     'POLAND':'POL',
                     'CZECHIA':'CZE',
                     'CROATIA':'HRV',
                     'SLOVAKIA':'SVK',
                     'FINLAND':'FIN',
                     'SLOVENIA':'SVN',
                     'GREECE':'GRC',
                     'SPAIN':'ESP',
                     'PORTUGAL':'PRT',
                     'NORWAY':'NOR'}

regions_dict = {'BELUX':'BEL-LUX',
                'BENELUX':'BEL-LUX-NLD',
                'CW_EU':'AUT-BEL-CHE-CZE-DEU-ESP-FRA-GBR-HRV-HUN-IRL-ITA-LUX-NLD-POL-PRT-SVK-SVN',
                'EU_GRP2':'AUT-BEL-CHE-DEU-DNK-FRA-GBR-IRL-ITA-LUX-NLD',
                'NW_EU':'BEL-DEU-DNK-FRA-GBR-IRL-LUX-NLD',
                'NW_EU2':'BEL-DEU-FRA-GBR-IRL-LUX-NLD',
                'NW_EU_CONTINENT':'BEL-DEU-FRA-LUX-NLD'}

regions_dict_old = {'CW_EU':'AUT-BEL-CHE-CZE-DEU-ESP-FRA-GBR-HRV-HUN-IRL-ITA-LUX-NLD-POL-PRT-SVK-SVK'}


countrycodes_dict.update(regions_dict)

# population from 2018 to 2023 (at Jan 1 each year)
bel_pop = np.array([11.399,11.455,11.522,11.555,11.618,11.723])
lux_pop = np.array([0.602,0.614,0.626,0.635,0.645,0.661])
bel_pop_r = np.round(np.mean(bel_pop/(bel_pop+lux_pop)),3)

def initialize_settings(ppt_mode: bool = False):
    """
    Extracts species and models info from json files.
    Defines standard colors for plotting.

    Args:
        ppt_mode (logical) (optional):
            If True, use bigger fonts (ideal for presentation slides)

    Returns:
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        model_colors (dict of lists):
            Default lists of colors to be used by each model.
        annotate_coords (dict of lists):
            Coordinates to annotate histogram.
    """

    # Read configuration files
    config_data = read_config_files()

    ### define colors

    model_colors = {'intem':[['blue','dodgerblue'],
                             ['dodgerblue','skyblue']],
                    'elris':[['purple','mediumpurple'],
                             ['deeppink','pink'],
                             ['darkorange','red']],
                    'rhime':[['darkgreen','green'],
                             ['limegreen','palegreen'],
                             ['olive','lightgreen']]}

    ### font settings & annotate_coords

    if (ppt_mode):
        plt.rc('font', size=15)
        plt.rc('axes', titlesize=18)
        plt.rc('axes', labelsize=16)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc('legend', fontsize=14)

        annotate_coords = {0:[0.58,0.65],
                           1:[0.58,0.40],
                           2:[0.58,0.15]}

        print('WARNING: Using big fonts. You might need to shrink the labels.')
    else:
        plt.rc('font', size=11)
        plt.rc('axes', titlesize=11)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)
        plt.rc('legend', fontsize=10)

        annotate_coords = {0:[0.6,0.80],
                           1:[0.6,0.60],
                           2:[0.6,0.40]}

    return config_data,model_colors,annotate_coords



def set_model_colors(models,model_colors):
    """
    Sets plotting colors for each model (updates model_colors).

    Args:
        models (list of str):
            Keys specifying model names, e.g. ['intem','elris']
        model_colors (dict of lists):
            Default lists of colors to be used by each model.

    Returns:
        mc (dict of lists):
            List of colors to be used by each model.
    """

    mc = dict()
    m0_list = np.unique([m.split('_')[0] for m in models])

    # If the different models result from a single inversion system
    if len(m0_list) == 1:
        inv_models = list(model_colors.keys())
        i = 0
        j = 0
        # Use model_colors in order
        for m in models:
            if j == len(model_colors[inv_models[i]]):
                i = i+1
                j = 0

            try:
                mc[m] = model_colors[inv_models[i]][j]
                j = j+1
            except:
                print('ERROR: Number of models is greater than number of colors in model_colors.')

    # If results from multiple inversion systems will be plotted together
    else:
        tmp_m0 = models[0].split('_')[0]
        j = 0
        for m in models:
            m0 = m.split('_')[0]
            if m0 != tmp_m0:
                tmp_m0 = m0
                j = 0

            try:
                mc[m] = model_colors[m0][j]
                j = j+1
            except:
                print(f'ERROR: Trying to use color number {j+1}, but there are only {j} colors defined for {m0}.')

    return mc
