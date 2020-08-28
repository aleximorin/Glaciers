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
from pyGM_funcs import *
import matplotlib.patches as patches
import os
from dataclasses import dataclass
from typing import List
from rasterio.mask import mask, geometry_mask

import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
import alphashape


def parse_scale_dict(dict):
    default_dict = {'x_offset': 0.05,
                    'y_offset': 0.05,
                    'units': 'km',
                    'length': 1,
                    'color': 'black'}

    for kw in dict.keys():
        default_dict[kw] = dict[kw]

    return default_dict


def array_to_table(array, row_name=None, get_header=False):
    # Mean, Std, Median, Maximum, Minimum, Mean absolute
    header = 'Mean, Std, Median, Maximum, Minimum, Mean absolute, RMSE'.split(', ')
    array = array[~np.isnan(array)]
    table = [array.mean(), array.std(), np.median(array), array.max(), array.min(), np.abs(array).mean(),
             np.sqrt(np.mean(array ** 2))]

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
        self.meta = None

        # Parameters to be computed and set through the associated Glacier object
        self.thickness = None
        self.thickness_array = None
        self.error = None
        self.rel_error = None
        self.error_array = None
        self.rel_error_array = None
        self.table = None
        self.point_extent = None

    def set_thickness(self, im):
        self.thickness = im

    def set_thickness_array(self, array):
        self.thickness_array = array

    def set_error(self, error_im, mask):
        self.error = error_im
        self.error_array = self.error[mask]

    def set_rel_error(self, rel_error_im, mask):
        self.rel_error = rel_error_im
        self.rel_error_array = self.rel_error[mask]

    def compute_statistics(self, get_header=False):
        # Statistics function.
        thickness_table = array_to_table(self.thickness, row_name='Thickness [m]', get_header=get_header)
        error_table = array_to_table(self.error, row_name='Error [m]', get_header=False)
        rel_error_table = array_to_table(self.rel_error, row_name='Relative error [%]', get_header=False)

        ymod = self.thickness_array
        y = self.thickness_array - self.error_array

        r2 = 1 - np.nansum((y - ymod) ** 2) / np.nansum((y - np.nanmean(y)) ** 2)

        table = [*thickness_table, error_table, rel_error_table, ['R2', r2]]
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

    def __init__(self, name: str, tag: str = '', dem_path: str = None, outline_path: str = None, gpr_path: any = None,
                 img_folder: str = '', delimiter: str = ',', whitespace: bool = False,
                 header: int = None, point_crs: str = None):

        # Basic parameters needed for saving the outputs
        self.name = name
        self.img_folder = img_folder + f'/{self.name}'
        self.tag = tag

        # Creates a pandas DataFrame tracking the various statistics of different models
        self.statistics = pd.DataFrame()

        # Initializes parameters to be changed if GPR data is given
        """ Wrong approach if the DEM is too high of a resolution, 
        can't upsample a poor resolution thickness model. Need corrections """
        self.true_thickness_im = None
        self.point_extent = None
        self.true_thickness_array = None

        # Creates the folder needed for the glacier's images
        try:
            os.makedirs(self.img_folder)
        except OSError:
            pass
        except Exception as e:
            print(e)

        # Opens the dem's raster
        self.dem = rasterio.open(dem_path, 'r')
        self.meta = self.dem.meta.copy()

        # Much simpler if the crs is from the DEM, no re-projection needed
        self.crs = self.dem.crs
        self.outline = gpd.read_file(outline_path)

        # Tries reading the gpr ice thickness data
        # Can either take as input a xyz formatted csv or a Pandas DataFrame
        if gpr_path is not None:
            self.set_gpr_data(gpr_path, delimiter=delimiter, whitespace=whitespace,
                              header=header, point_crs=point_crs)

            # Creates a rectangle to be used in various plots from the extent of the GPR data
            # Current bug where the rectangle can only be used in one plot.
            # Could be fixed but it is weird and shouldn't happen
        else:
            self.gpr = None

        # Ensures that the shapefile for the geometry has the same projection as the DEM
        if self.outline.crs != self.crs:
            self.outline = self.outline.to_crs(self.crs)

        # Crops the dem to the geometry of the shapefile
        # Gets the image of the dem that falls within the extent of the outline
        # Gets the metadata and the extent of the new image

        self.extent = rasterio.transform.array_bounds(self.dem.height, self.dem.width, self.dem.transform)
        self.dem_im = crop_raster_to_geometry(self.dem, self.outline.envelope)
        self.dem_im[self.dem_im <= 0] = np.NaN
        # A list of models, to be updated
        self.models = {}
        self.dem.close()

    def create_rectangle(self, color='black', lw=1):
        c_xmin, c_ymin, c_xmax, c_ymax = self.gpr.total_bounds
        rectangle = patches.Rectangle((c_xmin, c_ymin), c_xmax - c_xmin, c_ymax - c_ymin,
                                      linewidth=lw, edgecolor=color,
                                      facecolor='none')  # , label='Ground penetrating\nradar area')
        return rectangle

    def plot_transects_from_gpr_points(self, models, transects_path, field_name, merge=False, interp_field=None,
                                       point_dist=1000, text_color='black', showplot=False):

        if isinstance(transects_path, str):
            transects = gpd.read_file(transects_path)
        else:
            transects = transects_path

        @dataclass
        class Line:
            ref: str
            geometry: geom.MultiLineString

        # Merge lines having the same field
        if merge:
            field_values = np.sort(transects[field_name].unique())
            lines = [transects[transects[field_name] == i].unary_union for i in field_values]
            lines = [Line(i, j) for i, j in zip(field_values, lines)]
        else:
            lines = [Line(i, j) for i, j in zip(transects.index.to_list(), transects.geometry)]

        nrows = len(lines)
        ncols = 1
        for i in range(1, nrows):
            if nrows % i == 0:
                ncols = i

        nrows = int(nrows / ncols)

        fig, axs = plt.subplots(nrows, ncols)

        for line, ax in zip(lines, axs.flatten()):

            if isinstance(line.geometry, geom.MultiLineString):
                line.geometry = ops.linemerge(
                    [geom.LineString(tuple(map(tuple, np.round(subline.xy, 4).transpose()))) for subline in
                     line.geometry])

            for model in models:
                indices, points = indices_along_line(line.geometry, model.thickness.shape, model.meta)
                h = model.thickness[indices]
                dists = cumulative_distances(*points)
                ax.plot(dists, h, label=model.name)

            df = self.gpr[self.gpr[field_name] == line.ref].copy()
            x, y = df.iloc[:, :2].T.values
            df['dists'] = cumulative_distances(x, y)

            if interp_field is not None:
                # Puts nan values along the inferred data for the non inferred line and vice-versa
                inferred = df.iloc[:, 2].copy()
                inferred[df[interp_field] == 0] = np.NaN

                measured = df.iloc[:, 2].copy()
                measured[df[interp_field] == 1] = np.NaN

                ax.plot(df['dists'],
                        inferred,
                        label=f'Interpolated thickness', c='Red', linestyle='dotted')

                ax.plot(df['dists'],
                        measured,
                        label=f'Measured thickness', c='Black')
            else:
                ax.plot(df['dists'],
                        df.iloc[:, 2],
                        label=f'Measured thickness', c='Black')
            ax.set_title(line.ref)
            ax.invert_yaxis()

        if ncols != 1:
            plt.tight_layout(pad=3, h_pad=1, w_pad=1)
        else:
            plt.tight_layout()
        handles, labels = axs.flatten()[-1].get_legend_handles_labels()
        plt.legend(bbox_to_anchor=(0.5, -0), loc="lower center",
                   borderaxespad=0, ncol=len(labels), bbox_transform=fig.transFigure, frameon=False)

        fig2 = plt.figure()
        ax0 = plt.gca()
        self.outline.plot(ax=ax0, facecolor='None', edgecolor='black')
        self.plot_map(self.dem_im, ax=ax0, hillshade=True, alpha=0)

        for line in lines:
            n = int(np.floor(line.geometry.length / point_dist))
            points = [line.geometry.interpolate(i * point_dist) for i in range(n + 1)]
            x = [i.x for i in points]
            y = [i.y for i in points]
            """ u = np.diff(x)
            v = np.diff(y)
            pos_x = x[:-1]
            pos_y = y[:-1]
            norm = np.sqrt(u ** 2 + v ** 2)
            ax0.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", headwidth=1, width=0.005,
                       headlength=1, headaxislength=1, color='blue')"""
            plt.scatter(x, y, c='blue')

            x, y = line.geometry.xy
            ax0.plot(x, y, c='blue')

            X, Y = np.array(line.geometry.centroid.xy)
            label = line.ref
            ax0.text(X, Y, label, c=text_color)

        xmin, ymin, xmax, ymax = transects.total_bounds
        ax0.set_xlim(xmin, xmax)
        ax0.set_ylim(ymin, ymax)
        if showplot:
            plt.show()

        fig.savefig(f'{self.img_folder}/transects.png')
        fig2.savefig(f'{self.img_folder}/transects_map.png')

    def plot_transects_from_rasters(self, models, transects_path, field_name, merge=False, interp_field=None,
                                    point_dist=1000, text_color='black', showplot=False, simplify=None):

        if isinstance(transects_path, str):
            transects = gpd.read_file(transects_path)
        else:
            transects = transects_path

        @dataclass
        class Line:
            ref: str
            geometry: geom.MultiLineString

        # Merge lines having the same field
        if merge:
            field_values = np.sort(transects[field_name].unique())
            lines = [transects[transects[field_name] == i].unary_union for i in field_values]
            lines = [Line(i, j) for i, j in zip(field_values, lines)]
        else:
            lines = [Line(i, j) for i, j in zip(transects.index.to_list(), transects.geometry)]

        nrows = len(lines)
        ncols = 1
        for i in range(1, nrows):
            if nrows % i == 0:
                ncols = i

        nrows = int(nrows / ncols)

        fig, axs = plt.subplots(nrows, ncols)

        for line, ax in zip(lines, axs.flatten()):

            if isinstance(line.geometry, geom.MultiLineString):
                line.geometry = ops.linemerge(
                    [geom.LineString(tuple(map(tuple, np.round(subline.xy, 4).transpose()))) for subline in
                     line.geometry])

            if simplify is not None:
                line.geometry = line.geometry.simplify(simplify)

            for model in models:
                indices, points = indices_along_line(line.geometry, model.thickness.shape, model.meta)
                h = model.thickness[indices]
                dists = cumulative_distances(*points)
                ax.plot(dists, h, label=model.name)

            true_thickness = (model.thickness - model.error)[indices]
            ax.plot(dists, true_thickness, label='Measurements')
            ax.set_title(line.ref)
            ax.invert_yaxis()

        if ncols != 1:
            plt.tight_layout(pad=3, h_pad=1, w_pad=1)
        else:
            plt.tight_layout()
        handles, labels = axs.flatten()[-1].get_legend_handles_labels()
        plt.legend(bbox_to_anchor=(0.5, -0), loc="lower center",
                   borderaxespad=0, ncol=len(labels), bbox_transform=fig.transFigure, frameon=False)

        fig2 = plt.figure()
        ax0 = plt.gca()
        self.outline.plot(ax=ax0, facecolor='None', edgecolor='black')
        self.plot_map(self.dem_im, ax=ax0, hillshade=True, alpha=0)

        for line in lines:
            n = int(np.floor(line.geometry.length / point_dist))
            points = [line.geometry.interpolate(i * point_dist) for i in range(n + 1)]
            x = [i.x for i in points]
            y = [i.y for i in points]
            """ u = np.diff(x)
                v = np.diff(y)
                pos_x = x[:-1]
                pos_y = y[:-1]
                norm = np.sqrt(u ** 2 + v ** 2)
                ax0.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", headwidth=1, width=0.005,
                           headlength=1, headaxislength=1, color='blue')"""
            plt.scatter(x, y, c='blue')

            x, y = line.geometry.xy
            ax0.plot(x, y, c='blue')

            X, Y = np.array(line.geometry.centroid.xy)
            label = line.ref
            ax0.text(X, Y, label, c=text_color)

        xmin, ymin, xmax, ymax = transects.total_bounds
        ax0.set_xlim(xmin, xmax)
        ax0.set_ylim(ymin, ymax)
        if showplot:
            plt.show()

        fig.savefig(f'{self.img_folder}/transects_rasters.png')
        fig2.savefig(f'{self.img_folder}/transects_map_rasters.png')

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
            df = pd.read_csv(gpr_path, delimiter=delimiter, delim_whitespace=whitespace,
                             header=header, keep_default_na=False)
            self.gpr = gpd.GeoDataFrame(df, crs=point_crs,
                                        geometry=[geom.Point(xy) for xy in zip(df.iloc[:, 0], df.iloc[:, 1])])
        except ValueError:
            if isinstance(gpr_path, gpd.GeoDataFrame):
                self.gpr = gpr_path

            if isinstance(gpr_path, pd.DataFrame):
                df = gpr_path
                self.gpr = gpd.GeoDataFrame(df, crs=point_crs,
                                            geometry=[geom.Point(xy) for xy in zip(df.iloc[:, 0], df.iloc[:, 1])])
        except Exception as e:
            print("Could not read the csv or DataFrame for the GPR data\n")
            print(e)

        # Ensures the point geometry is the right one
        if point_crs is not None:
            self.gpr = self.gpr.to_crs(self.crs)

        self.point_extent = self.gpr.total_bounds
        self.ashape = self.outline.intersection(self.gpr.unary_union.convex_hull)

    def add_model(self, model):

        # Function to add models to the glacier's dictionary of models
        # Computes the model's errors and gives it the attributes.
        # See the compute_error function

        self.compute_error(model)
        self.models[model.tag] = model
        table = model.compute_statistics(get_header=True)
        self.statistics = self.statistics.append(table)
        self.statistics = self.statistics.append([''])

    def compute_error(self, model):

        # Glacier function to compute the error of a given Model.
        # Sets the Model's attributes

        # Opens the given model and compute its error
        model_ras = rasterio.open(model.model_path)
        im = model_ras.read()
        meta = model_ras.meta.copy()
        model.meta = meta
        true_thickness_im, point_extent = points_to_raster(self.gpr.geometry, self.gpr.iloc[:, 2],
                                                           meta)
        mask = ~np.isnan(true_thickness_im)

        # Crops the raster to the extent of the glacier
        im = crop_image_to_geometry(im, meta, self.outline.geometry)
        model_ras.close()

        # Computes the error and associates various data for the Model
        error = im - true_thickness_im
        ttim = true_thickness_im.copy()
        err = error.copy()
        err[ttim < 5] = np.nan
        ttim[ttim < 5] = np.nan
        rel_error = 100 * err / ttim
        model.thickness = im
        model.thickness_array = im[mask]

        # Need to set error_array differently, some places have nan values in the model
        model.set_error(error, mask)
        model.set_rel_error(rel_error, mask)

        model.point_extent = point_extent

    def glate_optimize(self, model: Model, save_output: str = None):

        # Applies a least square coefficient beta such that the quantity
        # q = || observed_data - beta * modelled_data || ** 2 is minimized

        """
        Inputs
        model: A model object referenced within the glacier
        save_output: Defaults to None. Can be indicated a path to save the raster.
        """

        x = model.thickness_array
        y = x - model.error_array

        x, y = x[~np.isnan(x)], y[~np.isnan(x)]
        x, y = x[~np.isnan(y)], y[~np.isnan(y)]

        beta, _, _, _ = np.linalg.lstsq(x[:, np.newaxis], y[:, np.newaxis], rcond=None)
        mod_im = (model.thickness * beta).astype(self.meta['dtype'])

        if save_output is not None:
            with rasterio.open(save_output, 'w+', **self.meta) as output:
                output.write(mod_im)

        return beta[0][0], mod_im

    def plot_map(self, im: np.array, title: str = '', cbar_unit: str = None, tag: str = None,
                 meta: dict = None, cmap: str = 'viridis', view_extent: np.array = None,
                 outline: bool = False, points: bool = False, point_color: bool = False,
                 rectangle: bool = False, labels: bool = False, ticks: bool = True,
                 scamap: bool = None, ax: plt.Axes = None, hillshade: bool = False,
                 scale_dict: dict = None, grid: bool = True, alpha: float = 1, showplot: bool = False,
                 sci: bool = False, figsize: tuple = None, ashape: bool = None):

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
        hillshade: Boolean value, defaults to False. Set for a hillshade effect, especially on DEMs
        scale: Dictionary, defaults to None. Parameters to add a scale to the map
        alpha: Float defaults to 1. Sets the transparency of the main map image
        showplot: Boolean, defaults to False. Set to true if you want to see the figure. Only pops up if
                  no ax object is provided.
        """
        # Gets the two last dimensions of the 3D array. Needed because rasters from rasterio are of (1, m, n) size
        if len(im.shape) == 3:
            im = im[0]

        if meta is None:
            meta = self.meta

        # Manages the extent array. The order needed here is different than given from the shapely format
        b = [0, 2, 1, 3]
        extent = [rasterio.transform.array_bounds(*im.shape, meta['transform'])[i] for i in b]
        xmin, xmax, ymin, ymax = [self.outline.total_bounds[i] for i in b]

        if view_extent is not None:
            view_extent = [view_extent[i] for i in b]
            xmin, xmax, ymin, ymax = view_extent

        fig = None
        if ax is None:
            fig = plt.figure(figsize=figsize, tight_layout=True)
            ax = plt.gca()

        # Plot the image and add a colorbar
        if scamap is None:
            norm = Normalize(np.nanmin(im), np.nanmax(im))
            scamap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        if hillshade:
            hills = im_to_hillshade(self.dem_im[0], 225, 40)
            ax.imshow(hills, extent=[self.extent[i] for i in b], cmap='Greys')

        img = ax.imshow(im, cmap=cmap, extent=extent, alpha=alpha)

        if cbar_unit is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size=0.2, pad=0.1)
            cbar = plt.colorbar(img, cax=cax, orientation='vertical')
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_title(f'{cbar_unit}')

        c = 'black'
        e = None
        if point_color:
            c = scamap.to_rgba(self.gpr.iloc[:, 2])
            e = 'black'

        # Plots various map accessories if asked
        if points:
            lw = 0.1
            if len(self.gpr) > 1000:
                lw = 0
            ax.scatter(self.gpr.geometry.x, self.gpr.geometry.y, c=c,
                       cmap=cmap, edgecolors=e, linewidths=lw)

        if outline:
            self.outline.plot(ax=ax, facecolor='None', edgecolor='black')

        if rectangle:
            rec = self.create_rectangle(color='red', lw=1)
            ax.add_patch(rec)
        if ashape:
            x, y = self.ashape.unary_union.exterior.xy
            ax.plot(x, y, color='red', lw=1)

        if scale_dict is not None:
            scale_dict = parse_scale_dict(scale_dict)

            y_offset = (ymax - ymin) * scale_dict['y_offset']
            x_offset = (xmax - xmin) * scale_dict['x_offset']

            length = scale_dict['length']
            label_length = length

            if scale_dict['units'] == 'km':
                length *= 1000

            xs = [xmin + x_offset, xmin + x_offset + length]
            ys = [ymin + y_offset, ymin + y_offset]

            bar = ax.plot(xs, ys, linewidth=5, color=scale_dict['color'])
            txt = ax.text(np.mean(xs), np.mean(ys) + y_offset * (1 + 0.05 / scale_dict['y_offset']),
                          f'{label_length} {scale_dict["units"]}',
                          ha='center', va='center', color=scale_dict['color'])
            # txt.set_path_effects([pe.withStroke(linewidth=5, foreground='w')])

        # Customises the map
        ax.set_title(f'{title}')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')

        if labels:
            ax.set_xlabel('Eastings [m]')
            ax.set_ylabel('Northing [m]')
        if ticks:
            if sci:
                ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
            else:
                ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks().tolist()])
                ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks().tolist()], rotation=-45)

        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        if grid:
            ax.grid()

        # Defines a custom legend for the outline due to a geopandas bug
        # Currently no legend, need to figure out something for every model and dem source
        """handles, labels = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='black', label=data['outline']))
        handles.append(matplotlib.patches.Patch (color='none', label=crs))
        ax.legend(handles=handles, bbox_to_anchor=(0.5, data['leg_pos']), loc="lower center",
                  bbox_transform=fig.transFigure, ncol=len(handles), frameon=True)"""

        if fig is not None:
            if showplot:
                plt.show()
            if tag is not None:
                fig.savefig(f'{self.img_folder}/{tag}.png', bbox_inches='tight')
            return fig, ax

        else:
            return img

    def maps_subplot(self, models, view_extent, figname=None, attribute='thickness', cmap='viridis',
                     vertical=False, subtitles=None, figsize=None, common_cbar: bool = True, cbar_unit='[m]',
                     showfig: bool = False):

        ims = [getattr(model, attribute) for model in models]
        # Multiple subplots function. Can be improved.
        N = int(len(models))
        n = N
        w = 1
        pad = None
        for i in range(1, N):
            if N % i == 0:
                w = i
                n = int(N / w)
        cbar_loc = 'right'
        cbar_or = 'vertical'
        scamap = None
        if common_cbar:
            norm = Normalize(np.nanmin(ims), np.nanmax(ims))
            scamap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar_unit = None

        if vertical:
            w, n = n, w
            pad = 1
            cbar_loc = 'bottom'
            cbar_or = 'horizontal'
        fig, axs = plt.subplots(w, n, figsize=figsize)
        for i in range(N):
            model = models[i]
            ax = axs[i]
            im = ims[i]
            meta = models[i].meta
            if subtitles is None:
                subtitle = model.name
            else:
                subtitle = subtitles[i]
            img = self.plot_map(im, subtitle, cbar_unit=cbar_unit,
                                view_extent=view_extent, outline=True, ax=ax,
                                ticks=False, cmap=cmap, meta=meta, hillshade=True, scamap=scamap)
        if common_cbar:
            for ax in axs:
                pad = 0
                if ax == axs[-1]:
                    pad = 0.05
                divider = make_axes_locatable(ax)
                cax = divider.append_axes(cbar_loc, size=0.15, pad=pad)
                cbar = plt.colorbar(img, cax=cax, orientation=cbar_or)
                cbar.ax.set_xlabel('[m]')
                if ax != axs[-1]:
                    cax.remove()

        plt.tight_layout()
        # fig.suptitle(f'{self.name}, modelled {attribute}')
        if figname is not None:
            fig.savefig(f'{self.img_folder}/{figname}.png', bbox_inches='tight')

        if showfig:
            plt.show()

    def plot_elevation(self, **kwargs):
        # Simple function call to plot the DEM
        self.plot_map(im=self.dem_im,
                      cbar_unit='[m asl]',
                      tag=f'elevation',
                      outline=True,
                      cmap='terrain',
                      alpha=0.7,
                      hillshade=True,
                      ashape=True,
                      ticks=True,
                      labels=True,
                      **kwargs)

    def scatterplot(self, model, title='', showplot=False,
                    figsize: tuple = None, ax=None, label=True, legend=True, nbins: int = 5,
                    same_aspect=True):
        # Plot the measured to modelled data of a given Model object
        fig = None
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        y = model.thickness_array
        x = y - model.error_array
        ax.scatter(x, y)
        x = [np.nanmin([x, y]), np.nanmax([x, y])]
        ax.plot(x, x, label='$x=y$ line', color='orange')

        if label:
            ax.set_xlabel('Measured thickness [m]')
            ax.set_ylabel('Modelled thickness [m]')
        ax.set_title(title)

        if legend:
            ax.legend()

        if same_aspect:
            ax.set_aspect('equal')

        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins))

        if fig is not None:
            if figsize is not None:
                fig.set_size_inches(figsize)
            plt.tight_layout()
            fig.savefig(f'{self.img_folder}/xy_scatter_{model.tag}.png', bbox_inches='tight')
            if showplot:
                plt.show()
            plt.close(fig)

    def boxplot(self, errors, labels, title='', yaxis='', tag=''):

        # Plots the error for every given model on a boxplot
        # errors is a list of np array
        fig = plt.figure()
        ax = plt.gca()
        plt.title(title)
        plt.ylabel(yaxis)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.boxplot(errors, labels=labels)
        plt.xticks(rotation=90)
        plt.tight_layout()
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

    def histogram(self, array, title='', xlabel='', tag='', showplot=False):
        # Plots a histogram for a given model
        fig = plt.figure()
        array = array[~np.isnan(array)]
        n = int(np.ceil(np.sqrt(len(array))))
        plt.title(title)
        plt.ylabel('N')
        plt.xlabel(xlabel)
        h, _, _ = plt.hist(array, n)
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        fig.savefig(f'{self.img_folder}/{tag}.png', bbox_inches='tight')
        if showplot:
            plt.show()
        plt.close(fig)

    def all_plots(self, model):

        # Main plotting function.
        scale_dict = {'length': 2, 'color': 'black'}
        print(f'Processing trough {model.tag}')
        # Plots thickness maps
        self.plot_map(model.thickness, f'Modelled thickness of {self.name}\n{model.name}', '[m]',
                      f'thickness_{model.tag}', outline=True, over_dem=True, scale_dict=scale_dict, grid=False,
                      ticks=False)
        self.plot_map(model.thickness, f'Modelled thickness of {self.name}\n{model.name}', '[m]',
                      f'thickness_cropped_{model.tag}', outline=True, points=True, point_color=True,
                      view_extent=self.point_extent, over_dem=True, scale_dict=scale_dict, grid=False,
                      ticks=False)

        # Plots error maps
        self.plot_map(model.error, f'Error\n{model.name}\n{self.name}', '[m]',
                      f'error_{model.tag}', outline=True,
                      view_extent=self.point_extent, scale_dict=scale_dict, grid=False,
                      ticks=False)
        self.plot_map(model.rel_error, f'Relative error\n{model.name}\n{self.name}', '[m]',
                      f'error_rel_{model.tag}', outline=True,
                      view_extent=self.point_extent, over_dem=True, scale_dict=scale_dict, grid=False,
                      ticks=False)

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
