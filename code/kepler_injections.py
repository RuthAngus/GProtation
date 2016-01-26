from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import glob

def inj(amin=1e-4, amax=5e-2):
    """
    load light curves of quiet stars and simulated light curves and add them
    together.
    amin, amax = minumum and maximum injection amplitudes.
    """

    # load ids of quiet stars
    ids = np.genfromtxt("data/quiet_kepler_ids.txt", dtype=int).T
    d = "simulations/kepler_injections"

    # load file names of simulations
    sim_fnames = np.sort(np.array(glob.glob("{0}/????_?????????.txt".format(d))))

    # generate list of amplitudes
    vmin, vmax = 1e-7, 1e-5
    nsim = len(sim_fnames)
    amps = np.exp(np.random.uniform(np.log(amin), np.log(amax), nsim))

    n = 0
    # match the light curves together
    for i, id in enumerate(ids):
        matches = np.array([fname.find(str(id)) for fname in sim_fnames])
        fnames = sim_fnames[matches > 0]  # find the right files to add together
        for j, fname in enumerate(np.sort(fnames)):  # add the lcs together
            print(fname)
            xsim, ysim = np.genfromtxt(fname).T
            x, y, yerr = np.genfromtxt("data/{0}_lc.txt".format(id)).T
            std = np.std(y)
            y, yerr = y / std, yerr / std
            new_y = y + amps[n] * ysim
            plt.clf()
            plt.plot(x, y, "k.")
            plt.plot(x, new_y, "g.")
            plt.plot(x, amps[n] * ysim, "r.")
            print("Var = ", np.var(amps[n] * ysim))
            fn = "simulations/kepler_injections"
            plt.savefig("{0}/{1}".format(fn, str(n).zfill(4)))
            dat = np.vstack((x, new_y , yerr))
            np.savetxt("{0}/{1}.txt".format(fn, str(n).zfill(4)), dat.T)

            n += 1

    data = np.vstack((np.arange(nsim), amps))
    np.savetxt("simulations/kepler_injections/true_amps.txt", data.T)

if __name__ == "__main__":
    inj(.1, 1)
