#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
from argparse import RawTextHelpFormatter

# local imports
from pyGM import Glacier, Model


def getparser():
    description = "Minimize mismatch between an inversion method output and GPR data using " \
                  "GlaTe Algorithm from Langhammer et al. (2019)"
    parser = argparse.ArgumentParser(
        description=description, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "tag",
        type=str,
        help="The glacier's tag which will be used when referencing it in files"
    )
    parser.add_argument(
        "-name",
        type=str,
        help="The glacier's name which will be used when referencing it in plots. "
             "(Optional, will use tag if no name is available.)"
    )
    parser.add_argument(
        "-src_dir", type=str, help="The source directory containing the input files"
    )
    parser.add_argument(
        "dem",
        type=str,
        help="path to surface DEM (if -src_dir given only need relative path)",
    )
    parser.add_argument(
        "shp",
        type=str,
        help="path to GLIMS outlines (if -src_dir given only need relative path)",
    )
    parser.add_argument(
        "gpr",
        type=str,
        help="path to gpr data in .xyz format (if -src_dir given only need relative path)",
    )
    parser.add_argument(
        "inv",
        type=str,
        help="path to inversion data being tuned (if -src_dir given only need relative path)",
    )
    parser.add_argument(
        "-plot", action="store_true", help="Generate generic output plots"
    )
    parser.add_argument(
        "-out_dir",
        type=str,
        help="Directory name to store output files (Does not need to exist, will be created)",
    )
    return parser


def main():
    parser = getparser()
    args = parser.parse_args()

    tag = args.tag
    name = args.name
    if not name:
        name = tag

    src_dir = args.src_dir

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = "glate_output"

    plots = args.plot

    # Generate file paths to input data
    if src_dir:
        # if src_dir is given make full paths
        dem_fp = os.path.join(src_dir, args.dem)
        shp_fp = os.path.join(src_dir, args.shp)
        gpr_fp = os.path.join(src_dir, args.gpr)
        inv_fp = os.path.join(src_dir, args.inv)
    else:
        dem_fp = args.dem
        shp_fp = args.shp
        gpr_fp = args.gpr
        inv_fp = args.inv

    # initiate a Glacier object with the kwags passed
    # GOI (Glacier of Interest)
    GOI = Glacier(
        name=name,
        tag=tag,
        dem_path=dem_fp,
        outline_path=shp_fp,
        gpr_path=gpr_fp,
        whitespace=True,
        header=None,
        img_folder=out_dir,
    )

    # create a Model object with the kwags passed
    GOI_inv = Model(
        model_name="Input Inversion",
        model_path=inv_fp,
        tag="input_inversion",
    )

    # add the inversion result to the glacier object
    GOI.add_model(GOI_inv)

    out_fp = os.path.split(os.path.splitext(inv_fp)[0])[-1] + '_glate.tif'
    glate_output = os.path.join(out_dir, out_fp)
    # error of the inversion result is minimized with the algorithm from L. Langhammer et al.:
    #   - https://doi.org/10.5194/tc-13-2189-2019
    alpha_gpr, opt_inv = GOI.glate_optimize(GOI_inv, save_output=glate_output)

    # create a a new Model object with the optimized inversion the glacier results
    GOI_inv_glate = Model(
        model_name="glate Optimized Inversion",
        model_path=glate_output,
        tag=GOI_inv.tag + "_glate",
    )

    # add the optimized inversion results to the glacier object
    GOI.add_model(GOI_inv_glate)

    # if -plots flag is passed then plots are generated
    if plots:
        # A plot of the glacier's surface DEM
        GOI.plot_elevation(title='Surface elevation')

        # A plot of the given inversion data
        GOI.plot_map(
            im=GOI_inv.thickness,
            title='Input Inversion Thickness',  # The title of the plot
            cbar_unit="[m]",  # The color bar's units
            tag=GOI_inv.tag,  # The plot's tag i.e. the name of the saved file
            outline=True,
            meta=GOI_inv.meta,
            hillshade=True,
            labels=True
        )

        # We can take a look at the new model
        GOI.plot_map(
            im=GOI_inv_glate.thickness,
            title=f"${alpha_gpr:2.2f} \cdot$ ({GOI_inv.name})",
            cbar_unit="[m]",
            tag=GOI_inv_glate.tag,
            outline=True,
            meta=GOI_inv_glate.meta,
            hillshade=True,
            labels=True
        )

        # A box plot can be made comparing our two models
        # Once again, see the documentation if needed, the functions are easily played with
        #GOI.plot_boxplots()


if __name__ == "__main__":
    main()
