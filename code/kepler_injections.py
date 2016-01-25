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

# generate list of amplitudes
vmin, vmax = 1e-7, 1e-5
amin, amax = .0001, .01
nsim = len(sim_fnames)
amps = np.exp(np.random.uniform(np.log(amin), np.log(amax), nsim))

r_kid, r_ind, r_a = [], [], []
n = 0
# match the light curves together
for i, id in enumerate(ids):
    matches = np.array([fname.find(str(id)) for fname in sim_fnames])
    fnames = sim_fnames[matches > 0]  # find the right files to add together
    for j, fname in enumerate(np.sort(fnames)):  # add the lcs together
        print(fname)
        xsim, ysim = np.genfromtxt(fname).T
        x, y, yerr = np.genfromtxt("data/{0}_lc.txt".format(id)).T
        new_y = y + amps[n] * ysim
        plt.clf()
        plt.plot(x, y, "k.")
        plt.plot(x, new_y, "g.")
        plt.plot(x, amps[n] * ysim, "r.")
        print("Var = ", np.var(amps[n] * ysim))
        fn = "simulations/kepler_injections"
        plt.savefig(str(n).zfill(4))
        dat = np.vstack((x, new_y , yerr))
        np.savetxt((str(n).zfill(4))

        r_kid.append(int(id))
        r_ind.append(j)
        r_a.append(amps[n])
        n += 1

data = np.vstack((np.array(r_kid), np.array(r_ind), np.array(r_a)))
np.savetxt("simulations/kepler_injections/true_amps.txt", data.T)
