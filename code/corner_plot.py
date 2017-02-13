import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 13,
           'ytick.labelsize': 13,
           'text.usetex': True}
plt.rcParams.update(plotpar)

id = 25
RESULTS_DIR = "/Users/ruthangus/projects/GProtation/code/results_01_24"
FIG_DIR = "/Users/ruthangus/projects/GProtation/documents/figures"
samples = pd.read_hdf
df = pd.read_hdf(os.path.join(RESULTS_DIR, "{}.h5".format(id)), key="samples")

truths = [None, None, None, None, np.log(20.8)]
labels = ["$\ln(A)$", "$\ln(l)$", "$\ln(\Gamma)$", "$\ln(\sigma)$",
          "$\ln(P)$"]
fig = corner.corner(df, labels=labels, truths=truths, range=[(-14.9, -14.1),
                                                             (6.3, 6.7),
                                                             (-.8, -.15),
                                                             (-19.15, -18.88),
                                                             (3, 3.07)])
fig.savefig(os.path.join(FIG_DIR, "corner_plot.pdf"))

# range (iterable (ndim,)) – A list where each element is either a length 2
# tuple containing lower and upper bounds or a float in range (0., 1.) giving
# the fraction of samples to include in bounds, e.g., [(0.,10.), (1.,5), 0.999,
# etc.]. If a fraction, the bounds are chosen to be equal-tailed.
