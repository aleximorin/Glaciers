from pyGM import Glacier, Model
from pyGM_funcs import *
import os

if __name__ == '__main__':
    path = 'pyGM_test_case'  # Path of the folder where the data is located on your hard drive
    folder = 'Glacier_plot_folder'  # Path of the folder where you want the

    try:
        os.makedirs(folder)
    except OSError as e:
        pass

    # We can easily instantiate a glacier by calling the Glacier object
    # You can refer to the documentation in the Glacier.py file for more specific information
    ng = Glacier('North Glacier', 'north_glacier',
                 dem_path=f'{path}/north_glacier_dem.tif',  # Path of the dem, works best with .tif
                 outline_path=f'{path}/north_glacier_utm.shp',  # Path of the outline
                 gpr_path=f'{path}/north_glacier_gpr.xyz',  # Path of the GPR data, can be txt or a DataFrame
                 whitespace=True, header=None, img_folder=folder)  # Some more parameters, see documentation

    # A plot of the glacier's surface DEM is easily made, the plot is saved but not shown
    ng.plot_elevation()

    # We can then create a Model object and associate it with the north_glacier Glacier object
    ng_consensus = Model(model_name='North Glacier, consensus, Farinotti (2019)',
                         model_path=f'{path}/ng_consensus.tif',
                         tag='farinotti_consensus')
    ng.add_model(ng_consensus)

    """
    The model is stored in a dictionary within the north_glacier Glacier object
    It can then be accessed in two ways:
        1. With the variable north_glacier_consensus
        2. Calling it from north_glacier.models['farinotti_consensus']
        
    The 2nd option is rather useful, meaning every model can be iterated on, as you would iterate on a dictionary
    """

    # Some information of the model can be easily plotted with built-in functions
    # See documentation of the Glacier.plot_map function
    ng.plot_map(im=ng_consensus.thickness,  # The 3D array to be plotted
                title=ng_consensus.name,  # The title of the plot
                cbar_unit='[m]',  # The color bar's units
                tag=ng_consensus.tag,  # The plot's tag i.e. the name of the saved file
                outline=True, points=True, point_color=True)  # Plotting parameters

    # The error of the model can be reduced according to the algorithm as made by L. Langhammer et al.:
    # https://doi.org/10.5194/tc-13-2189-2019
    # Note that you don't have to save a new .tif, see the documentation

    glate_output = f'{path}/glate_{ng_consensus.tag}.tif'
    glac_fac, mod_im = ng.glate_optimize(ng_consensus, save_output=glate_output)

    # This model can then be added to the dictionary of models like earlier
    ng_consensus_glate = Model(model_name='North Glacier, GlaTe optimized consensus',
                               model_path=glate_output,
                               tag='glate_farinotti_consensus')
    ng.add_model(ng_consensus_glate)

    # We can take a look at the new model
    ng.plot_map(im=ng_consensus_glate.thickness,  # The 3D array to be plotted
                title=f'${glac_fac:2.2f} \cdot {ng_consensus.name}$',  # The title of the plot
                cbar_unit='[m]',  # The color bar's units
                tag=ng_consensus_glate.tag,  # The plot's tag i.e. the name of the saved file
                outline=True, points=True, point_color=True)  # Plotting parameters

    # A box plot can be made comparing our two models
    # Once again, see the documentation if needed, the functions are easily played with
    ng.plot_boxplots()