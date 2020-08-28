# pyGM
Simple python library to easily compare glacier thickness data and models

Created by Alexi Morin, 2020, USRA summer project

The Glacier and Model objects are extensively described in the pyGM.py python script.

The dependencies are:
  - rasterio
  - geopandas
  - shapely
  - numpy
  - pandas
  - matplotlib

Some of those are probably dependant on other ones, I'm not particularly familiar with the necessary order of installation.

Simply download the whole package and follow along the instructions located in the Glaciers_test_case.py file

### Setting up a conda environment s

Assuming that you have [Anaconda](https://www.anaconda.com/products/individual) installed.

```bash
conda env create -f pyGM.yml
```

To start you conda environment run  

```bash
conda activate pyGM
```
Command Line Tool
Get help with command line utility by running python glate.py -h:

glate.py [-h] [-src_dir SRC_DIR] [-plot] [-out_dir OUT_DIR]
                dem shp gpr inv
Minimize mismatch between input inversion and GPR data using GlaTe Algorithm from Langhammer et al. (2019)
positional arguments:
  dem               path to surface DEM (if -src_dir given only need relative path)
  shp               path to GLIMS outlines (if -src_dir given only need relative path)
  gpr               path to gpr data in .xyz format (if -src_dir given only need relative path)
  inv               path to inversion data being tuned (if -src_dir given only need relative path)
optional arguments:
  -h, --help        show this help message and exit
  -src_dir SRC_DIR  The source directory containing the input files
  -plot             Generate generic output plots
  -out_dir OUT_DIR  Directory name to store output files (Does not need to exist, will be created)
Using the example directory Glacier_test_case a call of the command line utility would be:

cmd_line.py north_glacier_dem.tif north_glacier_utm.shp north_glacier_gpr.xyz ng_consensus.tif -src_dir pyGM_test_case
The output directory can be specified with with -out_dir flag. Also can save plotted data with the -plot flag:

cmd_line.py ng north_glacier_dem.tif north_glacier_utm.shp north_glacier_gpr.xyz ng_consensus.tif -src_dir pyGM_t
est_case -plot
