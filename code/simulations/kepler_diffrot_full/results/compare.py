from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from plotstuff import params, colours
reb = params()
cols = colours()

def compare_pgram(true_periods, ids, path):  # path is where results are saved

    # load recovered
    recovered_periods = np.zeros_like(ids)
    for i in range(len(ids)):
        id = str(int(ids[i])).zfill(4)
        recovered_periods[i], _ = \
                np.genfromtxt("{0}/{1}_pgram_result.txt".format(path, id)).T

    f = .3
    m = (recovered_periods < true_periods + true_periods*f) * \
            (true_periods - true_periods*f < recovered_periods)
    print("recovered", len(recovered_periods[m]), "out of ",
          len(recovered_periods),
          100*len(recovered_periods[m])/float(len(recovered_periods)), "%")

    plt.clf()
    recovered_periods[recovered_periods==-9999] = 0
    plt.plot(true_periods, recovered_periods, "^", color=cols.red,
             mec=cols.red, ms=3)
    m = clen < 3
#     plt.plot(true_periods[m], recovered_periods[m], "^", color=cols.blue,
#              mec=cols.blue, ms=5)
#     plt.ylim(0, 80)
    plt.xlabel("$\mathrm{True~P}_{\mathrm{rot}}~\mathrm{(days)}$")
    plt.ylabel("$\mathrm{Measured~P}_{\mathrm{rot}}~\mathrm{(days)}$")
    xs = np.linspace(min(true_periods), 100, 100)
    plt.plot(xs, xs, color=".5", ls="--")
    plt.xlim(0, 55)
    plt.savefig("pgram_compare_{0}.pdf".format(path))

    resids = ((recovered_periods - true_periods)**2)**.5
    print(max(resids))
    m = resids > 80
    print(ids[m])
    print("Pgram RMS = ", np.mean((recovered_periods - true_periods)**2)**.5)

def compare_acf(true_periods, ids, path):  # path is where results are saved

    # load recovered
    recovered_periods = np.zeros_like(ids)
    simple_periods = np.zeros_like(ids)
    errs = np.zeros_like(ids)
    for i in range(len(ids)):
        id = str(int(ids[i])).zfill(4)
        recovered_periods[i], errs[i] = \
                np.genfromtxt("{0}/{1}_acfresult.txt".format(path, id)).T
#         simple_periods[i], _ = \
#                 np.genfromtxt("{0}/{1}_simple_acfresult.txt".format(path,
#                               id)).T

    f = .3
    m = (recovered_periods < true_periods + true_periods*f) * \
            (true_periods - true_periods*f < recovered_periods)
    print("recovered", len(recovered_periods[m]), "out of ",
          len(recovered_periods),
          100*len(recovered_periods[m])/float(len(recovered_periods)), "%")

    plt.clf()
    recovered_periods[recovered_periods==-9999] = 0
    plt.plot(true_periods, recovered_periods, "s", color=cols.orange,
             mec=cols.orange, ms=3)
    plt.plot(true_periods, simple_periods, "s", color=cols.blue,
             mec=cols.orange, ms=3)
    plt.ylim(0, 80)
    plt.xlabel("$\mathrm{True~P}_{\mathrm{rot}}~\mathrm{(days)}$")
    plt.ylabel("$\mathrm{Measured~P}_{\mathrm{rot}}~\mathrm{(days)}$")
    xs = np.linspace(min(true_periods), 100, 100)
    plt.plot(xs, xs, color=".5", ls="--")
    plt.xlim(0, 55)
    plt.savefig("acf_compare_{0}.pdf".format(path))

    # plot a histogram of rotation periods
    plt.clf()
    plt.hist(true_periods, 12, histtype="stepfilled", color="w")
    plt.xlim(0, 55)
    plt.xlabel("$\mathrm{P}_{\mathrm{rot}}~\mathrm{(days)}$")
    plt.ylabel("$\mathrm{N}$")
    plt.savefig("acf_compare_{0}_hist.pdf".format(path))
    print("ACF RMS = ", np.mean((recovered_periods - true_periods)**2)**.5)

def compare_GP(true_periods, ids, path):

    # load recovered
    recovered_periods = np.zeros_like(ids)
    errps = np.zeros_like(ids)
    errms = np.zeros_like(ids)
    acf_periods, acf_errs = [np.zeros_like(ids) for i in range(2)]
    pgram_periods = np.zeros_like(ids)
    for i in range(len(ids)):
        id = str(int(ids[i])).zfill(4)
        fname = "{0}/{1}_samples.h5".format(path, id)
        if os.path.exists(fname):
            with h5py.File(fname, "r") as f:
                samples = f["samples"][...]
            nwalkers, nsteps, ndims = np.shape(samples)
            flat = np.reshape(samples, (nwalkers * nsteps, ndims))
            mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                              zip(*np.percentile(flat, [16, 50, 84], axis=0)))
            recovered_periods[i], errps[i], errms[i] = np.exp(mcmc_result[4])
        acf_periods[i], acf_errs[i] = \
                np.genfromtxt("{0}/{1}_acfresult.txt".format(path, id)).T
        pgram_periods[i], _ = \
                np.genfromtxt("{0}/{1}_pgram_result.txt".format(path, id)).T

    t = .1
    m = (acf_periods - t*acf_periods < pgram_periods) \
            * (pgram_periods < acf_periods + t*acf_periods)
    plt.clf()
    plt.errorbar(true_periods[m], recovered_periods[m], yerr=errps[m],
                 fmt="k.", capsize=0)
    plt.ylim(0, 80)
    xs = np.linspace(min(true_periods), max(true_periods), 100)
    plt.plot(xs, xs, "r--")
    plt.plot(xs, .5*xs, "r--")
    plt.savefig("GP_compare_{0}.pdf".format(path))

    resids = ((recovered_periods - true_periods)**2)**.5
    print(max(resids))
    m = resids > 800
    print(ids[m])

if __name__ == "__main__":

    # Load Suzanne's noise-free simulations
    data = np.genfromtxt("../par/final_table.txt", skip_header=1).T
    m = data[13] == 0  # just the stars without diffrot
    ids = data[0][m]
    true_periods = data[-3][m]
    clen = data[2][m]

#     compare_acf(true_periods, ids, "noise-free")
#     compare_acf(true_periods, ids, "noisy")
    compare_GP(true_periods, ids, "noise-free")
#     compare_acf(true_periods, ids, "noisy")
#     compare_pgram(true_periods, ids, "noisy")
#     compare_pgram(true_periods, ids, "noise-free")
