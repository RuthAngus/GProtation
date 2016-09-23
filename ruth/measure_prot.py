# measure the rotation period of a kepler star

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from GProtation import make_plot, lnprob
from Kepler_ACF import corr_run
import h5py
import sys
import os
import time
import emcee
from kepler_data import load_kepler_data
import glob

plotpar = {'axes.labelsize': 22,
           'font.size': 22,
           'legend.fontsize': 22,
           'xtick.labelsize': 22,
           'ytick.labelsize': 22,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def recover_injections(id, DATA_DIR, RESULTS_DIR, npts=10, cutoff=20,
                       plot_data=False, lower_lim=.8, upper_lim=1.2,
                       burnin=5000, run=10000, nwalkers=20):
    """
    run MCMC on each star, initialising with the ACF period.
    :param id:
        The kepler id of the star.
    :param DATA_DIR:
        The directory containing the light curves.
    :param RESULTS_DIR:
        The directory to save result files to.
    :param npts: (optional)
        The number of points per period to down-sample to. Default = 10.
    :param c: (optional)
        The number of rotation periods after which to cut off. Default = 20.
    :param plot_data: (optional)
        boolean. Plots the original and downsampled light curve if True.
    :param lower_lim: (optional)
        lower limit for period prior. Default = .8 * period.
    :param upper_lim: (optional)
        upper limit for period prior. Default = 1.2 * period.
    :param burnin: (optional)
        Number of burnin steps. Default = 5000.
    :param run: (optional)
        Number of run steps. Default = 10000
    :param nwalkers: (optional)
        Number of walkers. Default = 20
    """

    id = str(int(id)).zfill(9)

    # load lightcurve
    fnames = glob.glob(os.path.join(DATA_DIR,
                                    "{0}/kplr{0}-*_llc.fits".format(id)))
    print(fnames)
    x, y, yerr = load_kepler_data(fnames)

    # initialise with acf
    fname = os.path.join(RESULTS_DIR, "{0}_acf_result.txt".format(id))
    if os.path.exists(fname):
        p_init, err = np.genfromtxt(fname)
    else:
        p_init, err = corr_run(x, y, yerr, id, RESULTS_DIR)
        np.savetxt(os.path.join("{0}_acf_result.txt".format(id)),
                   np.array([p_init, err]).T)
    print("acf period, err = ", p_init, err)

    # Format data
    sub = int(p_init / npts * 48)  # 10 points per period
    ppd = 48. / sub
    ppp = ppd * p_init
    print("sub = ", sub, "points per day =", ppd, "points per period =",
          ppp)
    xsub, ysub, yerrsub = x[::sub], y[::sub], yerr[::sub]
    c = cutoff * p_init  # cutoff
    m = xsub < (xsub[0] + c)
    xb, yb, yerrb = xsub[m], ysub[m], yerrsub[m]

    # plot data
    if plot_data:
        plt.clf()
        m = x < (xsub[0] + c)
        plt.errorbar(x[m], y[m], yerr=yerr[m], fmt="k.", capsize=0)
        plt.errorbar(xb, yb, yerr=yerrb, fmt="r.", capsize=0)
        plt.savefig(os.path.join(RESULTS_DIR, "{0}_sub".format(id)))

    # prep MCMC
    plims = np.log([lower_lim*p_init, upper_lim*p_init])
    print("total number of points = ", len(xb))
    theta_init = np.log([np.exp(-5), np.exp(7), np.exp(.6), np.exp(-16),
                        p_init])
    ndim = len(theta_init)
    p0 = [theta_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (xb, yb, yerrb, plims)

    # time the lhf call
    start = time.time()
    print("lnprob = ", lnprob(theta_init, xb, yb, yerrb, plims))
    end = time.time()
    tm = end - start
    print("1 lhf call takes ", tm, "seconds")
    print("burn in will take", tm * nwalkers * burnin, "s")
    print("run will take", tm * nwalkers * run, "s")
    print("total = ", (tm * nwalkers * run + tm * nwalkers * burnin)/60,
          "mins")

    # run MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
    print("burning in...")
    start = time.time()
    p0, _, state = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    print("production run...")
    p0, _, state = sampler.run_mcmc(p0, run)
    end = time.time()
    print("actual time = ", (end - start)/60, "minutes")

    # save samples
    f = h5py.File(os.path.join(RESULTS_DIR, "{}_samples.h5".format(id)), "w")
    data = f.create_dataset("samples", np.shape(sampler.chain))
    data[:, :] = np.array(sampler.chain)
    f.close()

    # make various plots
    with h5py.File(os.path.join(RESULTS_DIR, "{}_samples.h5".format(id)),
                   "r") as f:
        samples = f["samples"][...]
    make_plot(samples, xb, yb, yerrb, id, RESULTS_DIR, traces=True, tri=True,
              prediction=True)

if __name__ == "__main__":
    DATA_DIR = "/Users/ruthangus/.kplr/data/lightcurves"
    RESULTS_DIR = "results"
    id = 4760478
    recover_injections(id, DATA_DIR, RESULTS_DIR, plot_data=True, burnin=2,
                       run=50, nwalkers=8)
