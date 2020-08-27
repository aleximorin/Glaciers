import numpy as np
import pandas as pd
import gdal
import geopandas as gpd
import shapely.geometry as geom
import rasterio
from rasterio.warp import reproject
from rasterio.mask import mask, geometry_mask
import scipy.interpolate as interp
import salem


# Plenty of more or less documented functions, from useful to not really.
# Most of the functions actually used are directly imported in the plot_maker.py file


def interpolate_im(im, meta, method='cubic', out_ras=None):
    im[im < 0] = np.nan
    points_indices = np.where(~np.isnan(im))
    z = im[np.where(~np.isnan(im))]
    x, y = get_points_from_index(points_indices, meta)

    h, w = im.shape
    x0, y0, = meta['transform'][2], meta['transform'][5]
    dx, dy = meta['transform'][0], meta['transform'][4]

    X, Y = np.linspace(x0, x0 + dx * w, w), np.linspace(y0, y0 + dy * h, h)
    X, Y = np.meshgrid(X, Y)

    interp_im = interp.griddata((x, y), z, (X, Y), method=method)

    if out_ras is not None:
        with rasterio.open(out_ras, 'w+', **meta) as out:
            out.write(interp_im.astype(meta['dtype']), 1)
    return interp_im


def get_index_from_points(point_coordinates, meta):
    x0, y0, = meta['transform'][2], meta['transform'][5]
    dx, dy = meta['transform'][0], meta['transform'][4]

    x, y = np.array(point_coordinates).transpose()
    x_index, y_index = np.floor((x - x0) / dx).astype('int'), np.floor((y - y0) / dy).astype('int')

    return y_index, x_index

def get_points_from_index(yx_index, meta):

    d = 0
    # Gets the important information from the metadata and gives them an actual meaning
    if type(meta) == salem.Grid:
        dx, dy = meta.dx, meta.dy
        x0, y0 = meta.x0, meta.y0
    else:
        x0, y0, = meta['transform'][2], meta['transform'][5]
        dx, dy = meta['transform'][0], meta['transform'][4]

    if len(yx_index) > 2:
        yx_index = yx_index[1:]
    y, x = np.array(yx_index)

    x = (x + 0.5) * dx + x0
    y = (y + 0.5) * dy + y0
    points = (x, y)

    return points


def meta_box(meta):

    dx, dy = meta['transform'][0], meta['transform'][4]
    x0, y0 = meta['transform'][2], meta['transform'][5]

    return dx, dy, x0, y0


def indices_along_line(linestring, im_shape, meta):

    # In case the im is three-dimensional

    # Gets every x and y coordinates constituting the line
    x, y = np.array(linestring.xy)
    x_indices, y_indices = [], []

    # Checking every line between two consecutive points
    for i in range(len(x) - 1):
        if i == 0:
            p0 = (x[i], y[i])
        else:
            p0 = p1

        p1 = (x[i + 1], y[i + 1])

        # Gets every index the line goes through

        line = geom.LineString([p0, p1])
        indices = np.array(
            np.where(geometry_mask([line], im_shape[-2:], meta['transform'], all_touched=False, invert=True))).transpose()

        # Computes the numbers of x and y cells traversed by the line
        y_index, x_index = get_index_from_points([p1, p0], meta=meta)
        dy, dx = y_index[-1] - y_index[0], x_index[-1] - x_index[0]

        # Checks whether the line is horizontal or vertical
        last, first = 0, 1
        if np.abs(dy) < np.abs(dx):
            last, first = first, last

        # Sorts the array accordingly
        if dy > 0:
            if dx >= 0:
                indices = indices[np.lexsort([indices[:, last], indices[:, first]])]
            else:
                indices[:, 1] = -indices[:, 1]
                indices = indices[np.lexsort([indices[:, last], indices[:, first]])]
        else:
            indices[:, 0] = -indices[:, 0]
            if dx >= 0:
                indices = indices[np.lexsort([indices[:, last], indices[:, first]])]

            else:
                indices[:, 1] = -indices[:, 1]
                indices = indices[np.lexsort([indices[:, last], indices[:, first]])]

        # Makes sure the indices are positive, sometimes we need to sort descending
        indices = np.abs(indices)[::-1]

        ys, xs = indices[1:, 0], indices[1:, 1]
        y_indices.extend(ys)
        x_indices.extend(xs)

    indices = (y_indices, x_indices)
    points = get_points_from_index(indices, meta)

   # points = [linestring.interpolate(linestring.project(point)) for point in points]

    if len(im_shape) > 2:
        indices = np.array([np.zeros_like(indices[0]), *indices])

    return tuple(indices), points

def indices_along_line_v2(linestring, im_shape, meta):

    # In case the im is three-dimensional
    dx, dy, x0, y0 = meta_box(meta)

    # Gets every x and y coordinates constituting the line
    x, y = np.array(linestring.xy)
    x_indices, y_indices = np.floor((x-x0)/dx).astype('int'), np.floor((y-y0)/dy).astype('int')

    indices = pd.DataFrame([y_indices, x_indices]).T.drop_duplicates()
    points = np.array([x, y]).T[indices.index.values].T
    indices = np.array(indices).T

    if len(im_shape) > 2:
        indices = np.array([np.zeros_like(indices[0]), *indices])

    return tuple(indices), points


def cumulative_distances(x, y):
    return np.array([0, *np.cumsum(np.sqrt([np.sum((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) for i in range(1, len(x))]))])


def patch_raster(void_dem_path, filler_dem_path, output_ras=None, no_data_values=np.NaN):

    with rasterio.open(void_dem_path, 'r') as void_dem_ras:
        void_dem_meta = void_dem_ras.meta.copy()
        void_im = void_dem_ras.read()

    with rasterio.open(filler_dem_path, 'r') as filler_dem_ras:
        resized_im, resized_meta = resize_ras_to_target(filler_dem_ras, void_dem_meta)

    void_im[void_im == no_data_values] = np.nan
    cond = np.isnan(void_im)

    void_im[cond] = resized_im[cond]

    if output_ras is not None:
        with rasterio.open(output_ras, 'w', **void_dem_meta) as out:
            out.write(void_im)

    return void_im


def im_to_hillshade(array, azimuth, alt_angle):
    # Generates hillshade from a 2d array
    # https://www.neonscience.org/create-hillshade-py
    azimuth = 360 - azimuth
    x, y = np.gradient(array)
    slope = np.pi/2 - np.arctan(np.sqrt(x**2 + y**2))
    aspect = np.arctan2(-x, y)
    azimuth_rad = azimuth*np.pi/180
    alt_angle_rad = alt_angle*np.pi/180

    shaded = np.sin(alt_angle_rad)*np.sin(slope) + \
             np.cos(alt_angle_rad)*np.cos(alt_angle_rad)*np.cos(slope)*np.cos((azimuth_rad - np.pi/2) - aspect)

    return 255*(shaded + 1)/2


def df_dict_to_excel(df_dict, path, header=True, index=True):
    # Outputs a dictionary of DataFrame to an excel file with the DataFrame's key as the sheet name
    writer = pd.ExcelWriter(path, engine='xlsxwriter')

    for tab_name, df in df_dict.items():
        df.to_excel(writer, sheet_name=tab_name, header=header, index=index)

    saved = False
    extension = path[path.rfind('.'):]
    xl_file = path[:-len(extension)]
    i = 1
    while not saved:
        try:
            writer.save()
            saved = True
        except PermissionError:
            print(f'{path} not accesible.')
            print(f'Saving at {xl_file}')


def crop_raster_to_target(orig_ras, meta, output_ras=None):
    # Gets the data from the target raster and
    # creates a box from it's extent to resize the original raster to
    out_z, out_h, out_w = meta['count'], meta['height'], meta['width']
    extent = rasterio.transform.array_bounds(out_h, out_w, meta['transform'])
    bbox = geom.box(*extent)
    gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=meta['crs'])

    # Crops the original raster
    cropped, t = mask(orig_ras, shapes=gdf.geometry, crop=True, nodata=np.NaN)
    cropped = cropped.astype(meta['dtype'])

    out_meta = meta.copy()
    out_meta['transform'] = t
    out_meta['width'] = cropped.shape[-1]
    out_meta['height'] = cropped.shape[-2]

    if output_ras is not None:
        with rasterio.open(output_ras, 'w+', **out_meta) as out:
            out.write(cropped)

    return cropped, out_meta


def resize_ras_to_target(orig_ras, meta, output_ras=None):

    cropped_im, cropped_meta = crop_raster_to_target(orig_ras, meta)

    out_im = np.zeros(shape=(cropped_im.shape)).astype(meta['dtype'])
    out_im[out_im == 0] = np.NaN
    out_im, t = reproject(cropped_im,
                          destination=out_im,
                          src_transform=cropped_meta['transform'],
                          dst_transform=meta['transform'],
                          src_crs=orig_ras.crs,
                          dst_crs=meta['crs'])

    if output_ras is not None:
        if output_ras.endswith('.tif'):
            meta['driver'] = 'GTiff'

        # Writes the new raster with the target raster's metadata
        out = rasterio.open(output_ras, 'w+', **meta)
        out.write(out_im)

    return out_im, meta


def crop_raster_to_geometry(raster, geometry):
    im = raster.read()
    mask = geometry_mask(geometry, im[0].shape, raster.transform, all_touched=False)
    im[:, mask] = np.NaN
    return im


def crop_image_to_geometry(im, metadata, geometry):
    h, w = im.shape[-2], im.shape[-1]
    mask = geometry_mask(geometry, (h, w), metadata['transform'])

    try:
        im[:, mask] = np.NaN
    except IndexError:
        im[mask] = np.NaN

    return im


def points_to_raster(points, z, meta, nodata=np.nan, invert=True, add_axis=True):

    import salem

    # Gets the x and y values of the points
    try:
        x, y = points
    except:
        try: x, y = points.x, points.y
        except Exception as e: print(e)

    # Gets the important information from the metadata and gives them an actual meaning
    if type(meta) == salem.Grid:
        c, w, h = 1, meta.nx, meta.ny
        dx, dy = meta.dx, meta.dy
        x0, y0 = meta.x0, meta.y0
    else:
        c, w, h = meta['count'], meta['width'], meta['height']
        dx, dy = meta['transform'][0], meta['transform'][4]
        x0, y0, = meta['transform'][2], meta['transform'][5]

    # Creates the bins from the DEM's extent
    xs, ys = np.arange(x0, x0 + (w + 1)*dx, dx)[:w+1], np.flip(np.arange(y0, y0 + (h + 1)*dy, dy))[:h+1]

    # Number of points per cell
    n, _, _ = np.histogram2d(y, x, bins=(ys, xs))
    n[n == 0] = np.NaN

    # Sum of values per cell and compute the mean value
    im, y_edges, x_edges = np.histogram2d(y, x, bins=(ys, xs), weights=z, normed=False)
    im /= n

    im[np.isnan(im)] = nodata

    if add_axis:
        im = im.reshape((c, h, w))

    if invert:
        # Flip the array so it fits with the map's format
        im = np.flip(im, axis=1)

    # Computes the new raster's extents
    y_index = (~np.isnan(im)).sum(-1) != 0
    x_index = (~np.isnan(im)).sum(-2) != 0

    xmin = x_edges[x_index.argmax()]
    ymin = y_edges[np.flip(y_index).argmax()]

    xmax = x_edges[-np.flip(x_index).argmax()-1]
    ymax = y_edges[-y_index.argmax()-1]

    bounds = np.array([xmin, ymin, xmax, ymax])

    return im, bounds


if __name__ == '__main__':
    """import matplotlib.pyplot as plt

    dx = meta['transform'].a
    dy = meta['transform'].e
    x0 = meta['transform'].c
    y0 = meta['transform'].f
    w = meta['width'] + 1
    h = meta['height'] + 1

    xmin, ymin, xmax, ymax = linestring.bounds

    xgrid = np.linspace(x0, x0 + dx * w, w)
    ygrid = np.linspace(y0, y0 + dy * h, h)

    # xgrid = xgrid[(xgrid < xmax)]# and (xgrid > xmin)]
    # ygrid = ygrid[(ygrid < ymax) and (ygrid > ymin)]

    plt.plot(points[0], points[1], label='cell points', marker='.')
    plt.plot(*linestring.xy, label='true line')
    ll = tuple(np.array(points).transpose())
    X, Y = np.array([linestring.interpolate(linestring.project(geom.Point(point))).xy for point in ll]).transpose()[0]
    plt.scatter(X, Y, label='nearest points')
    ax = plt.gca()
    ax.set_xticks(xgrid)
    ax.set_yticks(ygrid)
    plt.grid()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.legend()"""