import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.projections import PolarAxes

import mpl_toolkits.axisartist.grid_finder as gf
import mpl_toolkits.axisartist.floating_axes as fa

import numpy as np
from fluxy import config
import pandas as pd
from typing import Literal



class TaylorDiagram:
    """
    Taylor Diagram class for visualizing model performance.
    It compares standard deviation, correlation coefficient, and RMSE between observations and model predictions.
    """

    def __init__(
            self, 
            sd_obs, 
            fig=None, 
            position=(1, 1, 1), 
            markersize=100, 
            sd_range=(0, 1.5), 
            sd_unit='ppb'
    ):
        """
        Initialize the Taylor Diagram.

        Parameters:
        - sd_obs: Standard deviation of the reference dataset (scalar).
        - fig: Matplotlib figure instance (optional).
        - position: Three integers (nrows, ncols, index). The subplot will take the index position
                    on a grid with nrows rows and ncols columns.
        - label: Label for the reference dataset (default: '_').
        """   
        self.sd_obs = sd_obs  # Standard deviation of the reference
        self.markersize = markersize

        # Polar transformation for the Taylor diagram
        tr = PolarAxes.PolarTransform()

        # Correlation coefficient labels and positions
        rlocs = np.concatenate(((np.arange(11.0) / 10.0), [0.95, 0.99]))  # Correlation values
        tlocs = np.arccos(rlocs)  # Convert correlations to polar angles
        gl1 = gf.FixedLocator(tlocs)  # Position of ticks
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))  # Format tick labels

        # Define the grid helper with axis limits and labels
        self.smin = sd_range[0]  # Minimum standard deviation
        self.smax = sd_range[1]  # Maximum standard deviation
        gh = fa.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi / 2, self.smin, self.smax),  # (theta_min, theta_max, r_min, r_max)
            grid_locator1=gl1,
            tick_formatter1=tf1,
        )

        # Create figure and subplot
        if fig is None:
            fig = plt.figure()
        ax = fa.FloatingSubplot(fig, *position, grid_helper=gh)
        fig.add_subplot(ax)

        # Customize the axes
        # Correlation coefficient axis (top)
        ax.axis['top'].set_axis_direction('bottom')
        ax.axis['top'].label.set_text('Correlation coefficient')
        ax.axis['top'].toggle(ticklabels=True, label=True)
        ax.axis['top'].major_ticklabels.set_axis_direction('top')
        ax.axis['top'].label.set_axis_direction('top')

        # Standard deviation axis (left)
        ax.axis['left'].set_axis_direction('bottom')
        ax.axis['left'].label.set_text('Standard deviation [{}]'.format(sd_unit))
        ax.axis['left'].toggle(ticklabels=True, label=True)
        ax.axis['left'].major_ticklabels.set_axis_direction('bottom')
        ax.axis['left'].label.set_axis_direction('bottom')

        # Standard deviation axis (right)
        ax.axis['right'].set_axis_direction('top')
        ax.axis['right'].label.set_text('Standard deviation [{}]'.format(sd_unit))
        ax.axis['right'].toggle(ticklabels=True, label=True)
        ax.axis['right'].major_ticklabels.set_axis_direction('left')
        ax.axis['right'].label.set_axis_direction('top')

        # Hide bottom axis (not used)
        ax.axis['bottom'].set_visible(False)

        # Draw constant correlation lines
        for angle in tlocs:  # Use tlocs for exact positions of correlation labels
            x = [0, self.smax * np.cos(angle)]  # Line endpoints (x-coordinates)
            y = [0, self.smax * np.sin(angle)]  # Line endpoints (y-coordinates)
            ax.plot(x, y, color='gray', linestyle='--', linewidth=0.5)  # Dashed gray lines

        # Set main axes for plotting
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates

        # Add reference point for the observed standard deviation
        l = self.ax.scatter(
            [0], self.sd_obs, color='k', marker='*', s=self.markersize
        )  # Reference point
        t = np.linspace(0, np.pi / 2)  # Angles for sd_obs contour
        r = np.zeros_like(t) + self.sd_obs
        self.ax.plot(t, r, 'k--', label='_')  # Reference sd_obs line
        self.samplePoints = [l]  # Collect sample points for the legend


    def add_samples(
            self, 
            sds, 
            pearsons, 
            label, 
            *args, 
            **kwargs
        ):
        
        """
        Add a sample point to the Taylor diagram.

        Parameters:
        - sd: Standard deviation of the sample.
        - r: Correlation coefficient of the sample.
        - *args, **kwargs: Additional plotting parameters (e.g., color, marker).
        """
        self.ax.scatter(
            np.arccos(pearsons), sds, s=self.markersize, zorder=10, label=label, *args, **kwargs
        )  # Plot in polar coordinates


    def add_contours(
            self, 
            levels=10, 
            **kwargs
        ):

        """
        Add RMSE contours to the Taylor diagram.
        """

        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax),
            np.linspace(0, np.pi / 2),
        )
        xs = rs * np.cos(ts)
        ys = rs * np.sin(ts)
        rmse = np.sqrt(self.sd_obs**2 + rs**2 - 2 * self.sd_obs * rs * np.cos(ts))
        contours = self.ax.contour(xs, ys, rmse, levels=levels, **kwargs)
        return contours



def plot_stats_mf(
    stats: pd.DataFrame,
    stats_to_plot: list[str],
    species: str,
    model_colors: dict[str, str],
    model_labels: dict[str, str],
    config_data: dict[dict],
    mf_units_print: str,
    stats_type: str,
    stats_ylim: dict[list] = None,
    start_date: str = None,
    end_date: str = None,
) -> Figure:
    """
    Plots statistics for all sites, for all models.

    Args:
        stats (pandas.DataFrame):
            Statistical measures, for each site and for each model.
        stats_to_plot (list of str):
            Statistical measures to plot.
        species (str):
            Gas species, e.g. 'ch4'.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        model_labels (dict of str):
            Models and corresponding labels used to plot the stats.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        mf_units_print (str):
            Mole fraction units used in plots
        stats_type (str):
            Type of statistics to be plotted. Should be the same as used in call to stats_mf().
        stats_ylim (dict of lists):
            Limits for y-axis of individual statistic plots. Can be given for selected
            statistics only or passed as None for automatic axis range.
        start_date (str) and end_date (str):
            Dates used to title the plot.
    Returns:
        fig (figure):
            Plot showing each model's fit statistics, for each site.
    """

    models = np.unique(stats["model"].to_numpy())
    # make sure model_labels are in the correct order
    model_str = [model_labels[k] for k in models]
    # plot colors from model names
    colors = [model_colors[k][0] for k in models]

    # determine strings used for plot subtitle
    stats_str = stats_type
    mf_str = ""
    if stats_type not in ["prior", "posterior"]:
        mf_str = " above BC"
        stats_str = stats_type.split("_")[0]

    long_stats = pd.melt(stats, id_vars=["model", "site"], value_vars=stats_to_plot)
    nrows = len(stats_to_plot)
    fig, ax = plt.subplots(nrows, 1, figsize=(10, 3 * nrows), tight_layout=True)
    for i, stat in enumerate(stats_to_plot):
        df_this_stats = long_stats[long_stats["variable"] == stat].pivot(
            index="site", columns="model", values="value"
        )

        df_this_stats.plot(
            kind="bar",
            ax=ax[i],
            stacked=False,
            color=colors,
            xlabel="",
            legend=False,
            zorder=3,
        )
        ax[i].grid(zorder=0)
        if stats_ylim is not None:
            if stat in stats_ylim.keys():
                ax[i].set_ylim(stats_ylim[stat][0], stats_ylim[stat][1])

        ylabel = config.stat_labels[stat]
        if stat not in ["pearson", "nrmse", "nn"]:
            ylabel = ylabel + " (" + mf_units_print + ")"
        ax[i].set_ylabel(ylabel)

    leg = ax[0].legend(
        ncol=3,
        borderpad=0.2,
        columnspacing=1.0,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        labels=model_str,
    )

    species_info = config_data["species_info"][species]
    fig.suptitle(
        (
            f'{species_info["species_print"]} {stats_str} model performance versus mole fraction observations{mf_str}'
            f"\n{start_date} to {end_date}"
        )
    )

    return fig


def plot_taylor_diagram(
    stats: dict[Literal['prior', 'posterior'], pd.DataFrame],
    model_colors: dict[str, str],
    model_labels: dict[str, str],
    plot_type_model: Literal['separate', 'together'] = 'separate',
    plot_type_stat: Literal['separate', 'together'] = 'separate',
    include: list[str] = ['prior', 'posterior'],
    sd_range: tuple[float, float] = (0, 2.5),
    sd_unit: str = 'ppb',
) -> Figure:
    """
    Plots statistics for all sites, for all models.

    Args:
        stats (pandas.DataFrame):
            Statistical measures, for each site and for each model.
        stats_to_plot (list of str):
            Statistical measures to plot.
        species (str):
            Gas species, e.g. 'ch4'.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        model_labels (dict of str):
            Models and corresponding labels used to plot the stats.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        mf_units_print (str):
            Mole fraction units used in plots
        stats_type (str):
            Type of statistics to be plotted. Should be the same as used in call to stats_mf().
        stats_ylim (dict of lists):
            Limits for y-axis of individual statistic plots. Can be given for selected
            statistics only or passed as None for automatic axis range.
        start_date (str) and end_date (str):
            Dates used to title the plot.
    Returns:
        fig (figure):
            Plot showing each model's fit statistics, for each site.
    """

    models = np.unique(stats[include[0]]['model'].to_numpy())

    # Set the number of rows and columns
    NCOLS_MAX = 3
    ncols = nrows = 1

    if plot_type_model != 'together' or plot_type_stat != 'together':

        if plot_type_model == 'separate' and plot_type_stat == 'separate':
            nsubplots = (len(models) * len(stats))

        elif plot_type_model == 'together' and plot_type_stat == 'separate':
            nsubplots = len(stats)

        elif plot_type_model == 'separate' and plot_type_stat == 'together':
            nsubplots = len(models)

        ncols = NCOLS_MAX if nsubplots >= NCOLS_MAX else nsubplots
        nrows = nsubplots // ncols + 1
    
    # Create the figure
    fig = plt.figure(figsize=(6 * ncols, 6 * nrows), tight_layout=True)

    # Create a dictionary to save positions of diagrams
    dict_diags_pos = {}

    # Loop over statistics (i.e., prior, posterior)
    for i, (s, stat) in enumerate(stats.items()):
        long_stat = pd.melt(stat, id_vars=['model', 'site'], value_vars=['pearson', 'sd_obs', 'sd_sim'])
        df_pearson = long_stat[long_stat['variable'] == 'pearson'].pivot(index='site', columns='model', values='value')
        df_sds_obs = long_stat[long_stat['variable'] == 'sd_obs'].pivot(index='site', columns='model', values='value')
        df_sds_sim = long_stat[long_stat['variable'] == 'sd_sim'].pivot(index='site', columns='model', values='value')

        # Loop over models
        for j, m in enumerate(models):
            
            # Get the position of the subplot depending on the case
            if plot_type_model == 'separate' and plot_type_stat == 'separate':
                index_pos = i * ncols + j + 1

            elif plot_type_model == 'together' and plot_type_stat == 'together':
                index_pos = 1

            elif plot_type_model == 'together' and plot_type_stat == 'separate':
                index_pos = i + 1

            elif plot_type_model == 'separate' and plot_type_stat == 'together':
                index_pos = j + 1

            # Get the statistics for the model m
            list_sds_obs = df_sds_obs[m].values
            list_sds_sim = df_sds_sim[m].values
            list_pearsons = df_pearson[m].values

            # Create the Taylor diagram using the corresponding class or fetch it
            if index_pos not in dict_diags_pos:
                diag = TaylorDiagram(
                    list_sds_obs[0], 
                    fig=fig, 
                    position=(nrows, ncols, index_pos),
                    markersize=150, 
                    sd_range=sd_range, 
                    sd_unit=sd_unit
                )

            else:
                diag = dict_diags_pos[index_pos]

            # Add RMSE contours
            # plt.clabel(diag.add_contours(colors='0.5'), inline=1, fontsize=10)

            # Define labels, colors and markers
            label = model_labels[m] + ' - ' + s
            color = model_colors[m][0] if s == 'prior' else model_colors[m][1] 
            marker = 'o' if s == 'prior' else 's'

            # Add samples to the diagram
            diag.add_samples(list_sds_sim, list_pearsons, c=color, edgecolor='k', marker=marker, label=label)
            diag.ax.legend()

            # Save the position of the diagram's subplot
            dict_diags_pos[index_pos] = diag

    return fig