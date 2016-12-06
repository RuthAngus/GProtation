import matplotlib.pyplot as plt
import numpy as np

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)    

import os
import matplotlib.pyplot as plt
import numpy as np

from emcee3.backends import HDFBackend

from .model import GPRotModel, GPRotModel2

def trace_plot(star, directory='mcmc_chains', kepler=False, thin=10):
    filename = os.path.join(directory, '{}.h5'.format(star))
    b = HDFBackend(filename)
    coords = b.get_coords()
    ndim = coords.shape[-1]
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8,8))
    for i,ax in enumerate(axes):
        ax.plot(coords[::thin, :, i], lw=1, alpha=0.2);
        if i==ndim - 1:
            from .aigrain import AigrainTruths
            truth = AigrainTruths().df
            ax.axhline(np.log(truth.ix[star, 'PEQ']), lw=1, color='r')        
        ax.set_ylabel(GPRotModel.param_names[i])
            
    axes[0].set_title(star)
    return fig    

