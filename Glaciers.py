import rasterio
import geopandas as gpd
import pandas as pd
import shapely.geometry as geom
import shapely.ops as ops
from rasterio.mask import mask
from rasterio import features
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from inversion.qgis_functions import *
import matplotlib.patches as patches
import os
from dataclasses import dataclass
from typing import List
from rasterio.mask import mask, geometry_mask


def array_to_table(array, row_name=None, get_header=False):
    # Mean, Std, Median, Maximum, Minimum, Mean absolute
    header = 'Mean, Std, Median, Maximum, Minimum, Mean absolute'.split(', ')
    array = array[~np.isnan(array)]
    table = [array.mean(), array.std(), np.median(array), array.max(), array.min(), np.abs(array).mean()]

    if row_name is not None:
        table = [row_name, *table]

    if get_header:
        table = (header, table)

    return table


class Model:

    # Model class.

    """ Inputs
    model_name: A string which will be used in the plots
    model_path: The path of the model's .tif
    tag: A tag which will be use for saving the plots produced
    """

    def __init__(self, model_name: str, model_path: str, tag: str):
        self.name = model_name
        self.model_path = model_path
        self.tag = tag

        # Parameters to be computed and set through the associated Glacier object
        self.thickness = None
        self.thickness_array = None
        self.error = None
        self.rel_error = None
        self.error_array = None
        self.rel_error_array = None
        self.table = None

    def set_thickness(self, im):
        self.thickness = im

    def set_thickness_array(self, array):
        self.thickness_array = array

    def set_error(self, error_im):
        self.error = error_im
        self.error_array = self.error[~np.isnan(self.error)]

    def set_rel_error(self, rel_error_im):
        self.rel_error = rel_error_im
        self.rel_error_array = self.rel_error[~np.isnan(self.rel_error)]

    def compute_statistics(self, get_header=False):
        # Statistics function.
        thickness_table = array_to_table(self.thickness, row_name='Thickness [m]', get_header=get_header)
        error_table = array_to_table(self.error, row_name='Error [m]', get_header=get_header)
        table = [*thickness_table, error_table[-1]]
        table[0] = [self.name, *table[0]]
        self.table = table
        return table


class Glacier:

    # the main Glacier class. Can take as input a surface DEM, an outline and GPR data. Can store information for
    # multiple ice thickness models to be compared

    # Multiple functions can be used for plotting or comparing data

    """ Inputs
    name: The name of the glacier, will be used for the plots
    tag: The glacier's tag if wanted. Shorter and simpler version of name
    dem_path: Where the dem's .tif is located
    outline_path: Where the outline's .shp is located
    gpr_path: Where the gpr data is located. Can be a csv or a pandas DataFrame. See the set_gpr_data function
    img_folder: Folder path where a folder for the specific glacier will be created.
    delimiter: Delimiter of the gpr csv file
    whitespace: If the csv file is delimited by inconsistent whitespaces instead of commas, defaults to False
    header: If the csv file has a header or not, defaults to None. Set to the index of the needed row. (Often 0)
    point_crs: If the gpr data has a different coordinate system that of the DEM
    """

    def __init__(self, name: str, tag: str = '', dem_path: str = None, outline_path: str = None, gpr_path: str = None,
                 img_folder: str = '', delimiter: str = ',', whitespace: bool = False,
                 header: str = None, point_crs: str = None):

        # Basic parameters needed for saving the outputs
        self.name = name
        self.img_folder = img_folder + f'/{self.name}'
        self.tag = tag

        # Creates a pandas DataFrame tracking the various statistics of different models
        self.statistics = pd.DataFrame()

        # Creates the folder needed for the glacier's images
        try:
            os.makedirs(self.img_folder)
        except OSError:
            pass
        except Exception as e:
            print(e)

        # Opens the dem's raster
        self.dem = rasterio.open(dem_path, 'r')

        # Much simpler if the crs is from the DEM, no re-projection needed
        self.crs = self.dem.crs

        # Tries reading the gpr ice thickness data
        # Can either take as input a xyz formatted csv or a Pandas DataFrame
        if gpr_path is not None:
            self.set_gpr_data(gpr_path, delimiter=delimiter, whitespace=whitespace, header=header, point_crs=point_crs)
        else:
            self.gpr = None

        # Ensures that the shapefile for the geometry has the same projection as the DEM
        self.outline = gpd.read_file(outline_path)
        if self.outline.crs != self.crs:
            self.outline = self.outline.to_crs(self.crs)

        # Crops the dem to the geometry of the shapefile
        # Gets the image of the dem that falls within the extent of the outline
        # Gets the metadata and the extent of the new image
        self.meta = self.dem.meta.copy()
        self.extent = rasterio.transform.array_bounds(self.dem.height, self.dem.width, self.dem.transform)
        self.dem_im = crop_raster_to_geometry(self.dem, self.outline.envelope)

        # Creates a rectangle to be used in various plots from the extent of the GPR data
        # Current bug where the rectangle can only be used in one plot.
        # Could be fixed but it is weird and shouldn't happen
        c_xmin, c_ymin, c_xmax, c_ymax = self.gpr.total_bounds
        self.rectangle = patches.Rectangle((c_xmin, c_ymin), c_xmax - c_xmin, c_ymax - c_ymin,
                                           linewidth=2, edgecolor='black',
                                           facecolor='none')  # , label='Ground penetrating\nradar area')

        # Computes a true thickness image to the shape of the dem
        self.true_thickness_im, self.point_extent = points_to_raster(self.gpr, self.gpr.iloc[:, 2], self.meta)
        self.true_thickness_array = self.true_thickness_im[~np.isnan(self.true_thickness_im)]

        # A list of models, to be updated
        self.models = {}
        self.dem.close()

    def plot_transects(self, transect_path, models, true_bed_df, merge_field=None, vertical=True,
                       north_south=True, east_west=True, plot_surface=True, plot_outline=False):

        # Still need to figure out the plots
        # Still need to plot the true bed

        @dataclass
        class Line:
            ref: str
            geometry: geom.MultiLineString

        transects = gpd.read_file(transect_path)

        # Grabs the important information
        x0, y0, = self.meta['transform'][2], self.meta['transform'][5]
        dx, dy = self.meta['transform'][0], self.meta['transform'][4]

        # Merge lines having the same field
        if merge_field is not None:
            field_values = np.sort(transects[merge_field].unique())
            # field_values = ['NA']
            lines = [transects[transects[merge_field] == i].unary_union for i in field_values]
            lines = [Line(i, j) for i, j in zip(field_values, lines)]
        else:
            merge_field = 'id'
            lines = [i for i in transects.geometry]
            field_values = np.arange(len(lines))
            lines = [Line(i, j) for i, j in zip(field_values, lines)]

        nbr_of_cols = len(lines)
        nbr_of_rows = 1

        for i in range(1, len(lines)):
            if nbr_of_cols % i == 0:
                nbr_of_rows = i
                nbr_of_cols = int(len(lines) / nbr_of_rows)

        i0 = 0
        j0 = 0

        if ~vertical:
            nbr_of_cols, nbr_of_rows = nbr_of_rows, nbr_of_cols

        h, l = nbr_of_rows, nbr_of_cols

        fig = plt.figure()
        fig.suptitle = f'Transects for {self.name}'

        if plot_outline:
            i0 = nbr_of_cols
            nbr_of_cols *= 2

            if vertical:
                i0, j0 = j0, i0
                h, l = l, h
                nbr_of_rows, nbr_of_cols = nbr_of_cols, nbr_of_rows

        if vertical:
            leg_index = (nbr_of_rows + 1, nbr_of_cols)

        else:
            leg_index = (nbr_of_rows, nbr_of_cols + 1)

        grid = plt.GridSpec(*leg_index)

        if plot_outline:
            outline_axs = grid[:h, :l]
            ax0 = fig.add_subplot(outline_axs)
            self.outline.plot(ax=ax0, facecolor='None', edgecolor='black')
            transects.plot(ax=ax0)

            for line in lines:
                x, y = np.array(line.geometry.centroid.xy)
                label = line.ref
                ax0.text(x, y, label)

            xmin, ymin, xmax, ymax = transects.total_bounds
            ax0.set_xlim(xmin, xmax)
            ax0.set_ylim(ymin, ymax)

        axs = []
        for j in range(j0, nbr_of_rows):
            for i in range(i0, nbr_of_cols):
                axs.append(fig.add_subplot(grid[j, i]))

        axs = np.array(axs)
        cols = [i for i in models]
        true_bed_tag = 'True thickness'
        cols = cols + [true_bed_tag]

        areas = pd.DataFrame(columns=cols, index=field_values)

        for line, ax in zip(lines, axs.flatten()):

            true_thickness = true_bed_df[true_bed_df[merge_field] == line.ref].copy()  # Are they already N-S, E-W?
            df_coords = true_thickness[['x', 'y']]
            true_thickness['Distances'] = [0,
                                           *np.cumsum([np.sqrt(np.sum((df_coords.iloc[i] - df_coords.iloc[i - 1]) ** 2))
                                                       for i in range(1, len(df_coords))])]

            x, y = df_coords.iloc[:, 0], df_coords.iloc[:, 1]
            # Gets the index of the x and y values in the thickness model
            x_index, y_index = np.floor((x - x0) / dx).astype('int'), np.floor((y - y0) / dy).astype('int')
            indices = np.array([(j, i) for i, j in zip(x_index, y_index)])

            # Gets the surface line in case we need to use it
            surface = self.dem_im[0, indices[:, 0], indices[:, 1]]

            # Computes the gate's area in km^2
            areas[true_bed_tag][line.ref] = np.trapz(true_thickness['Thickness'],
                                                     true_thickness['Distances']) * 10 ** -6

            # Polishes the plot
            ax.set_title(f'{line.ref}')
            ax.invert_yaxis()

            # Inverts the plot's y_axis if the surface is there.
            if plot_surface:
                true_thickness['Thickness'] = surface - true_thickness['Thickness']
                ax.invert_yaxis()
                ax.plot(true_thickness['Distances'], surface, label=f'Surface')

            inferred = true_thickness[['Distances', 'Thickness', 'Inferred']].copy()
            inferred.loc[inferred['Inferred'] == 0, 'Thickness'] = np.NaN

            measured = true_thickness[['Distances', 'Thickness', 'Inferred']].copy()
            measured.loc[measured['Inferred'] == 1, 'Thickness'] = np.NaN

            ax.plot(inferred['Distances'],
                    inferred['Thickness'],
                    label=f'Interpolated thickness', c='Red', linestyle='dotted')

            ax.plot(measured['Distances'],
                    measured['Thickness'],
                    label=f'Measured thickness', c='Black')

            # Gets the thickness values from every model
            for model in models.values():

                im = model.thickness
                thicknesses = im[0, indices[:, 0], indices[:, 1]]
                # thick_df = pd.DataFrame([true_thickness['Distances'].values, thicknesses]).transpose().iloc[pd.DataFrame(indices).drop_duplicates().index]

                # Will plot the bed elevation instead of the thickness
                if plot_surface:
                    thicknesses = surface - thicknesses

                ax.plot(true_thickness['Distances'], thicknesses, label=f'{model.name}')

                # Computes the area by integrating the thicknesses over the distance
                dist = true_thickness['Distances'][~np.isnan(thicknesses)]
                thicknesses = thicknesses[~np.isnan(thicknesses)]
                area = np.trapz(thicknesses, dist) * 10 ** -6  # km^2
                areas[model.tag][line.ref] = area

        # axs[-1].legend(loc='right', bbox_to_anchor=(1.08, 0.5), bbox_transform=fig.transFigure)
        handles, labels = axs[-1].get_legend_handles_labels()

        leg_ax = fig.add_subplot(grid[int(leg_index[0] / 2), leg_index[1] - 1])
        leg_ax.set_axis_off()

        if vertical:
            ncol = len(labels)

            for i in range(len(labels), 1):
                if len(labels) % i == 0:
                    ncol = i
            leg_ax.legend(handles, labels, loc='lower center', ncol=ncol)
        else:
            leg_ax.legend(handles, labels,
                          loc='center right')  # , bbox_to_anchor=(1, 0.5), bbox_transform=fig.transFigure)

        size = 2
        h_factor = 1
        w_factor = 1
        if plot_outline:
            if vertical:
                h_factor = 1 / 1.6

        else:
            if ~vertical:
                w_factor = nbr_of_cols / size

        fig.set_size_inches(nbr_of_cols * size * w_factor, nbr_of_rows * size * h_factor)
        # fig.canvas.draw()
        fig.tight_layout()

        areas['data-raw/data'] = (areas['True thickness'] - areas['consensus']) / areas['True thickness']
        areas['data-raw/raw'] = (areas['True thickness'] - areas['consensus']) / areas['consensus']

        # Saves the figure and the dataframe
        fig.savefig(f'{self.img_folder}/transects_{merge_field}_v2.png')
        areas.to_csv(f'{self.img_folder}/transects_area_{merge_field}_v2.csv')

    def compute_volume(self, models, shape_path=None, shape_name='', save_df=False):

        # Function to compute the volume of a given area or the whole extent of the glacier.
        # Outputs a DataFrame and saves the data to a csv file if asked

        # If there is a shape_path provided
        if shape_path is not None:
            shape = gpd.read_file(shape_path)

        # Computes the area
        dA = abs(self.meta['transform'][0] * self.meta['transform'][4])
        df = pd.DataFrame(columns=['Volume [km^3]'], index=[model.name for model in models.values()])

        for model in models.values():

            if shape_path is not None:
                thickness = crop_image_to_geometry(model.thickness, self.meta, shape.geometry)
            else:
                thickness = model.thickness

            # Computes the ice volume and puts it into a DataFrame
            thickness = thickness[~np.isnan(thickness)]
            volume = dA * thickness.sum() * 10 ** -9  # km^3
            df.loc()[model.name] = volume

        if save_df:
            # Saves the DataFrame to a csv
            df.to_csv(f'{self.img_folder}/volumes_{shape_name}.csv')

        return df

    def set_gpr_data(self, gpr_path, delimiter, whitespace, header, point_crs):
        # Reads the gpr ice thickness data
        # Can either take as input a formatted csv or a pandas DataFrame of the form:
        # X, Y, Ice thickness
        try:
            df = pd.read_csv(gpr_path, delimiter=delimiter, delim_whitespace=whitespace, header=header)
        except ValueError:
            df = gpr_path
        except Exception as e:
            print("Could not read the csv or DataFrame for the GPR data\n")
            print(e)

        # Ensures the point geometry is the right one
        if point_crs is not None:
            self.gpr = gpd.GeoDataFrame(df, crs=point_crs,
                                        geometry=[geom.Point(xy) for xy in zip(df.iloc[:, 0], df.iloc[:, 1])])
            self.gpr = self.gpr.to_crs(self.crs)
        else:
            self.gpr = gpd.GeoDataFrame(df, crs=self.crs,
                                        geometry=[geom.Point(xy) for xy in zip(df.iloc[:, 0], df.iloc[:, 1])])

    def add_model(self, model):

        # Function to add models to the glacier's dictionary of models
        # Computes the model's errors and gives it the attributes.
        # See the compute_error function

        self.compute_error(model)
        self.models[model.tag] = model
        table = model.compute_statistics(get_header=True)
        self.statistics.append(table)
        self.statistics.append([''])

    def compute_error(self, model):

        # Glacier function to compute the error of a given Model.
        # Sets the Model's attributes

        # Opens the given model and compute its error
        model_ras = rasterio.open(model.model_path)

        # Checks if the size of the pixels is the same
        if model_ras.transform[0] != self.dem.transform[0]:
            resized_ras = f'{self.img_folder}/{model.tag}_resized.tif'

            # Checks if there's already a created raster for this one
            try:
                model_ras = rasterio.open(resized_ras, 'r')

            # Creates the correct raster, could change the exception message
            except Exception as e:
                model_ras = resize_ras_to_target(model_ras, self.meta, resized_ras)

        # Crops the raster to the extent of the glacier
        im = crop_raster_to_geometry(model_ras, self.outline.geometry)
        model_ras.close()

        # Computes the error and associates various data for the Model
        error = im - self.true_thickness_im
        rel_error = 100 * error / self.true_thickness_im

        model.set_thickness(im)
        model.set_thickness_array(im[~np.isnan(self.true_thickness_im)])
        model.set_error(error)
        model.set_rel_error(rel_error)

    def glate_optimize(self, model: Model, save_output: str = None):

        # Applies a least square coefficient beta such that
        # q = || observed_data - beta * modelled_data || ** 2 is reduced

        """ Inputs
        model: A model object referenced within the glacier
        save_output: Defaults to None. Can be indicated a path to save the raster.
        """

        y = self.true_thickness_array
        x = model.thickness_array

        x, y = x[~np.isnan(x)], y[~np.isnan(x)]
        x, y = x[~np.isnan(y)], y[~np.isnan(y)]
        beta, _, _, _ = np.linalg.lstsq(x[:, np.newaxis], y[:, np.newaxis], rcond=None)
        mod_im = (model.thickness*beta).astype(self.meta['dtype'])

        if save_output is not None:
            with rasterio.open(save_output, 'w+', **self.meta) as output:
                output.write(mod_im)

        return beta[0][0], mod_im

    def plot_map(self, im: np.array, title: str, cbar_unit: str, tag: str, cmap: str = 'viridis',
                 view_extent: np.array = None, outline: bool = False, points: bool = False, point_color: bool = False,
                 rectangle: bool = False, labels: bool = False, ticks: bool = True, scamap=None, ax=None):

        # Main plotting function. Ensures that all other plots have the same parameters

        """ Inputs:
        im: The 3D np array to be plotted. Can be the DEM, the thickness, the error, etc.
        title: The title of the plot
        cbar_unit: The units of the color bar
        tag: The tag of the plot with which it will be saved
        cmap: The colormap wanted for the plot, defaults to viridis
        view_extent: The extent wanted for the plot. Defaults to None which will set the extent to the whole map.
                     Takes as input a numpy array, like the self.point_extent array
        Outline: Boolean value, defaults to False. Set to true if you want the outline plotted on the map.
        points: Boolean value, defaults to False. Set to true if you want the points plotted on the map.
        point_color: Boolean value, defaults to False. Set to true if you want the points colored by the thickness.
                     Could be changed in the future to a scalar map instead of a boolean.
        rectangle: Boolean value, defaults to False. Set to true if you want a rectangle outlining the points' extent
                   in the map. CURRENTLY DOESN'T WORK PROPERLY
        labels: Boolean value, defaults to False. Set to true if you want the x and y labels plotted.
        ticks: Boolean value, defaults to True. Set to true if you want the x and y ticks plotted.
        scamap: Scalarmap object, defaults to None. Set if you want a specific colormap scale. Useful for subplots.
        ax: matplotlib Ax object, defaults to None. Set if you want to specify on which ax to plot.
            Useful for subplots.
        """

        # Manages the extent array. The order needed here is different than given from the shapely format
        b = [0, 2, 1, 3]
        extent = [self.extent[i] for i in b]
        xmin, xmax, ymin, ymax = extent

        if view_extent is not None:
            view_extent = [view_extent[i] for i in b]
            xmin, xmax, ymin, ymax = view_extent

        # Gets the two last dimensions of the 3D array. Needed because rasters from rasterio are of (1, m, n) size
        im = im[0]

        # fullplot is a variable needed to track if the plot is part of a subplot or not
        fullplot = 0

        if ax is None:
            fullplot = 1
            fig = plt.figure()
            ax = plt.gca()

        # Plot the image and add a colorbar
        if scamap is None:
            norm = Normalize(np.nanmin(im), np.nanmax(im))
            scamap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        img = ax.imshow(im, cmap=cmap, extent=extent)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.2, pad=0.1)
        cbar = plt.colorbar(img, cax=cax, orientation='vertical')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_title(f'{cbar_unit}')

        # Plots various map accessories if asked
        if points and point_color:
            ax.scatter(self.gpr.geometry.x, self.gpr.geometry.y, c=scamap.to_rgba(self.gpr.iloc[:, 2]),
                       cmap=cmap)
        elif points:
            self.gpr.plot(ax=ax, markersize=0.5)  # , label=data['points']

        if outline:
            self.outline.plot(ax=ax, facecolor='None', edgecolor='black')

        if rectangle:
            ax.add_patch(self.rectangle)

        # Customises the map
        ax.set_title(f'{title}')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        if labels:
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
        if ticks:
            ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks().tolist()])
            ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks().tolist()])
            ax.tick_params(axis='x', labelrotation=-45)
            ax.tick_params(axis='y')
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        ax.grid()

        # Defines a custom legend for the outline due to a geopandas bug
        # Currently no legend, need to figure out something for every model and dem source
        """handles, labels = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='black', label=data['outline']))
        handles.append(matplotlib.patches.Patch(color='none', label=crs))
        ax.legend(handles=handles, bbox_to_anchor=(0.5, data['leg_pos']), loc="lower center",
                  bbox_transform=fig.transFigure, ncol=len(handles), frameon=True)"""

        if fullplot:
            fig.savefig(f'{self.img_folder}/{tag}.png', bbox_inches='tight')
            plt.close(fig)
            return

        return img

    def maps_subplot(self, models, view_extent, figname, attribute='thickness', cmap='viridis',
                     vertical=False, subtitles=None):

        # Multiple subplots function. Can be improved.

        N = int(len(models))
        n = N
        w = 1
        pad = None
        for i in range(1, N):
            if N % i == 0:
                w = i
                n = int(N / w)
        if vertical:
            w, n = n, w
            pad = 1

        fig, axs = plt.subplots(w, n)
        max = np.max([np.nanmax(i) for i in getattr(models[i], attribute)])
        min = np.min([np.nanmin(i) for i in getattr(models[i], attribute)])
        norm = Normalize(min, max)
        scamap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        for i in range(N):
            model = models[i]
            ax = axs[i]
            im = getattr(model, attribute)
            if subtitles is None:
                subtitle = model.name
            else:
                subtitle = subtitles[i]
            img = self.plot_map(im, subtitle, '[m]', '',
                                view_extent=view_extent, outline=True, ax=ax, ticks=False, cmap=cmap)
        """cbar = plt.colorbar(img)
        cbar.ax.set_title('[m]')"""
        plt.tight_layout()
        # fig.suptitle(f'{self.name}, modelled {attribute}')
        fig.savefig(f'{self.img_folder}/{figname}.png', bbox_inches='tight')

    def plot_elevation(self):
        # Simple function call to plot the DEM
        self.plot_map(self.dem_im, f'Surface elevation\n{self.name}',
                      '[m]', 'elevation', 'terrain', outline=True, points=True, rectangle=True)

    def scatterplot(self, model):
        # Plot the measured to modelled data of a given Model object
        fig = plt.figure()
        plt.scatter(self.true_thickness_array, model.thickness_array)
        x = [np.min(self.true_thickness_array), np.max(self.true_thickness_array)]
        plt.plot(x, x, label='$x=y$ line', color='orange')
        plt.xlabel('Measured thickness [m]')
        plt.ylabel('Modelled thickness [m]')
        plt.title(f'Measured and modelled thickness, {model.name} for {self.name}')
        plt.legend()
        fig.savefig(f'{self.img_folder}/xy_scatter_{model.tag}.png', bbox_inches='tight')
        plt.close(fig)

    def boxplot(self, errors, labels, title, yaxis, tag):

        # Plots the error for every given model on a boxplot
        # errors is a list of np array
        fig = plt.figure()
        ax = plt.gca()
        plt.title(title)
        plt.ylabel(yaxis)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.boxplot(errors, labels=labels)
        plt.xticks(rotation=90)
        fig.savefig(f'{self.img_folder}/{tag}.png', bbox_inches='tight')
        plt.close(fig)

    def plot_boxplots(self, models=None):

        if models is None:
            errors = [model.error_array for model in self.models.values()]
            rel_errors = [model.rel_error_array for model in self.models.values()]
            labels = [f'{model.name}' for model in self.models.values()]
        else:
            labels = [f'{model.name}' for model in models.values()]
            rel_errors = [model.rel_error_array for model in models.values()]
            errors = [model.error_array for model in models.values()]

        self.boxplot(errors, labels,
                     f'Ice thickness error for {self.name}',
                     'Error [m]', 'boxplot')

        self.boxplot(rel_errors, labels,
                     f'Relative ice thickness error for {self.name}',
                     'Error [%]', 'boxplot_rel')

    def histogram(self, array, title, xlabel, tag):
        # Plots a histogram for a given model
        fig = plt.figure()
        array = array[~np.isnan(array)]
        n = int(np.ceil(np.sqrt(len(array))))
        plt.title(title)
        plt.ylabel('N')
        plt.xlabel(xlabel)
        plt.hist(array, n)
        fig.savefig(f'{self.img_folder}/{tag}.png', bbox_inches='tight')
        plt.close(fig)

    def all_plots(self, model):

        # Main plotting function.

        print(f'Processing trough {model.tag}')
        # Plots thickness maps
        self.plot_map(model.thickness, f'Modelled thickness of {self.name}\n{model.name}', '[m]',
                      f'thickness_{model.tag}', outline=True)
        self.plot_map(model.thickness, f'Modelled thickness of {self.name}\n{model.name}', '[m]',
                      f'thickness_cropped_{model.tag}', outline=True, points=True, point_color=True,
                      view_extent=self.point_extent)

        # Plots error maps
        self.plot_map(model.error, f'Error\n{model.name}\n{self.name}', '[m]',
                      f'error_{model.tag}', outline=True,
                      view_extent=self.point_extent)
        self.plot_map(model.rel_error, f'Relative error\n{model.name}\n{self.name}', '[m]',
                      f'error_rel_{model.tag}', outline=True,
                      view_extent=self.point_extent)

        # Plots scatter and histogram
        self.scatterplot(model)
        self.histogram(model.thickness, f'Ice thickness histogram of {model.name}',
                       'Ice thickness [m]', f'hist_thickness_{model.tag}')
        self.histogram(model.error_array, f'Error histogram of {model.name}',
                       'Error [m]', f'hist_error_{model.tag}')

    def all_models(self):
        # Main loop function
        for model in self.models.values():
            self.all_plots(model)
