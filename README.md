# Glaciers
Simple python library to easily compare glacier thickness data and models

Created by Alexi Morin, 2020, USRA summer project

The Glacier and Model objects are extensively described in the Glaciers.py python script.

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
conda env create -f Glacier.yml
```

To start you conda environment run  

```bash
conda activate Glacier
```
