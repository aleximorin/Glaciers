import numpy as np
import pandas as pd
import gdal
import geopandas as gpd
import shapely.geometry as geom
import rasterio
from rasterio.warp import reproject
from rasterio.mask import mask, geometry_mask


# Plenty of more or less documented functions, from useful to not really.
# Most of the functions actually used are directly imported in the compare_model_v2.py file

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

def resize_ras_to_target(orig_ras, meta, output_ras):

    # Gets the data from the target raster and
    # creates a box from it's extent to resize the original raster to
    out_z, out_h, out_w = meta['count'], meta['height'], meta['width']
    extent = rasterio.transform.array_bounds(out_h, out_w, meta['transform'])
    bbox = geom.box(extent[0], extent[1], extent[2], extent[3])
    gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=meta['crs'])

    # Crops the original raster
    cropped, t = mask(orig_ras, shapes=gdf.geometry, crop=True, nodata=np.NaN)
    cropped = cropped.astype(meta['dtype'])

    out_im = np.zeros(shape=(out_z, out_h, out_w)).astype(meta['dtype'])
    out_im[out_im == 0] = np.NaN
    out_im, t = reproject(cropped, destination=out_im, src_transform=t, dst_transform=meta['transform'],
                          src_crs=orig_ras.crs, dst_crs=orig_ras.crs)

    if output_ras.endswith('.tif'):
        meta['driver'] = 'GTiff'

    # Writes the new raster with the target raster's metadata
    out = rasterio.open(output_ras, 'w+', **meta)
    out.write(out_im)
    return out


def crop_raster_to_geometry(raster, geometry):
    im = raster.read()
    mask = geometry_mask(geometry, im[0].shape, raster.transform, all_touched=False)
    im[:, mask] = np.NaN
    return im


def crop_image_to_geometry(im, metadata, geometry):
    h, w = im.shape[-2], im.shape[-1]
    mask = geometry_mask(geometry, (h, w), metadata['transform'])
    im[:, mask] = np.NaN
    return im


def points_to_raster(points_shp, z, meta):

    # Gets the x and y values of the points
    x, y = points_shp.geometry.x, points_shp.geometry.y

    # Gets the important information from the metadata and gives them an actual meaning
    c, w, h = meta['count'], meta['width'], meta['height']
    dx, dy = meta['transform'][0], meta['transform'][4]
    x0, y0, = meta['transform'][2], meta['transform'][5]

    # Creates the bins from the DEM's extent
    xs, ys = np.arange(x0, x0 + (w+1)*dx, dx), np.flip(np.arange(y0, y0 + (h+1)*dy, dy))

    # Number of points per cell
    n, _, _ = np.histogram2d(y, x, bins=(ys, xs))
    n[n == 0] = np.NaN

    # Sum of values per cell and compute the mean value
    im, y_edges, x_edges = np.histogram2d(y, x, bins=(ys, xs), weights=z, normed=False)
    im /= n

    # Flip the array so it fits with the map's format
    im = np.flip(im.reshape((c, h, w)), axis=1)

    # Computes the new raster's extents
    y_index = (~np.isnan(im)).sum(2) != 0
    x_index = (~np.isnan(im)).sum(1) != 0

    xmin = x_edges[x_index.argmax()]
    ymin = y_edges[np.flip(y_index).argmax()]

    xmax = x_edges[-np.flip(x_index).argmax()-1]
    ymax = y_edges[-y_index.argmax()-1]

    bounds = np.array([xmin, ymin, xmax, ymax])

    return im, bounds
