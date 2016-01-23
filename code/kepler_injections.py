from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import glob

# This script loads light curves of quiet stars, loads the simulated light
# curves and adds them together

# load ids of quiet stars
ids = np.genfromtxt("data/quiet_kepler_ids.txt", dtype=int).T
d = "simulations/kepler_injections"

# load file names of simulations
sim_fnames = np.array(glob.glob("{0}/????_?????????.txt".format(d)))

# match the light curves together
for i, id in enumerate(ids):
    matches = np.array([fname.find(str(id)) for fname in sim_fnames])
    fnames = sim_fnames[matches > 0]  # find the right files to add together
    for j, fname in enumerate(fnames):  # add the lcs together
        xsim, ysim = np.genfromtxt(fname).T
        x, y, yerr = np.genfromtxt("data/{0}_lc.txt".format(id)).T
        plt.clf()
#         plt.plot(x, y + ysim, "g.")
#         plt.plot(x, ysim, "r.")
        plt.plot(x, y, "k.")
        plt.savefig("{0}_add_{1}".format(str(j).zfill(4), str(id).zfill(9)))
        print("{0}_add_{1}".format(str(j).zfill(4), str(id).zfill(9)))
        assert 0
