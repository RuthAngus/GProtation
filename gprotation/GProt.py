# This script contains a function for running the MCMC.

# coding: utf-8
from __future__ import print_function
import numpy as np
from GProtation import make_plot, lnprob
from Kepler_ACF import corr_run
import h5py
from gatspy.periodic import LombScargle
import os
import time
import emcee
import pyfits
import matplotlib.pyplot as plt

DATA_DIR = "data/"
RESULTS_DIR = "results/"


def load_k2_data(epic_id, DATA_DIR):
    hdulist = \
        pyfits.open(os.path.join(
            DATA_DIR,
            "hlsp_everest_k2_llc_{0}-c04_kepler_v1.0_lc.fits"
            .format(epic_id)))
    time, flux = hdulist[1].data["TIME"], hdulist[1].data["FLUX"]
    out = hdulist[1].data["OUTLIER"]
    m = np.isfinite(time) * np.isfinite(flux) * (out < 1)
    med = np.median(flux[m])
    return time[m], flux[m]/med - 1


def calc_p_init(x, y, yerr, id):

    print("Calculating ACF")
    fname = os.path.join(RESULTS_DIR, "{0}_acf_result.txt".format(id))
    print(fname)
    if os.path.exists(fname):
        print("Previous ACF result found")
        acf_period, err = np.genfromtxt(fname).T
    else:
        acf_period, err, lags, acf = corr_run(x, y, yerr, id, RESULTS_DIR)
        if not os.path.exists(fname):
            np.savetxt(fname, np.transpose((acf_period, err)))
    print("acf period, err = ", acf_period, err)

    fname = os.path.join(RESULTS_DIR, "{0}_pgram_result.txt".format(id))
    print(fname)
    if os.path.exists(fname):
        print("Previous pgram result found")
        pgram_period, pgram_period_err = np.genfromtxt(fname).T
    else:
        print("Calculating periodogram")
        ps = np.arange(.1, 100, .1)
        print(type(x), type(y), type(yerr))
        model = LombScargle().fit(x, y, yerr)
        pgram = model.periodogram(ps)

        plt.clf()
        plt.plot(ps, pgram)
        plt.savefig(os.path.join(RESULTS_DIR, "{0}_pgram".format(id)))
        print("saving figure ", os.path.join(RESULTS_DIR,
                                             "{0}_pgram".format(id)))

        peaks = np.array([i for i in range(1, len(ps)-1) if pgram[i-1] <
                          pgram[i] and pgram[i+1] < pgram[i]])
        pgram_period = ps[pgram == max(pgram[peaks])][0]
        print("pgram period = ", pgram_period, "days")
        pgram_period_err = pgram_period * .1
        np.savetxt(fname, np.transpose((pgram_period, pgram_period_err)))

    return acf_period, err, pgram_period, pgram_period_err


def fit(x, y, yerr, p_init, id, burnin=500, nwalkers=12, nruns=10,
        full_run=500):
    """
    Run the MCMC
    """

    plims = np.log([.1*p_init, 5*p_init])
    print("total number of points = ", len(x))
    theta_init = np.log([np.exp(-5), np.exp(7), np.exp(.6), np.exp(-16),
                         p_init])
    runs = np.zeros(nruns) + full_run
    ndim = len(theta_init)
    p0 = [theta_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (x, y, yerr, plims)

    # Time the LHF call.
    start = time.time()
    print("lnprob = ", lnprob(theta_init, x, y, yerr, plims))
    end = time.time()
    tm = end - start
    print("1 lhf call takes ", tm, "seconds")
    print("burn in will take", tm * nwalkers * burnin, "s")
    print("each run will take", tm * nwalkers * runs[0]/60, "mins")
    print("total = ", (tm * nwalkers * np.sum(runs) + tm * nwalkers *
                       burnin)/60, "mins")

    # Run MCMC.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
    print("burning in...")
    p0, _, state, prob = sampler.run_mcmc(p0, burnin)

    sample_array = np.zeros((nwalkers, sum(runs), ndim))
    for i, run in enumerate(runs):
        print("run {0} of {1}".format(i, len(runs)))
        sampler.reset()
        print("production run, {0} steps".format(int(run)))
        start = time.time()
        p0, _, state, prob = sampler.run_mcmc(p0, run)
        end = time.time()
        print("time taken = ", (end - start)/60, "minutes")

        # save samples
        sample_array[:, sum(runs[:i]):sum(runs[:(i+1)]), :] = \
            np.array(sampler.chain)
        f = h5py.File(os.path.join(RESULTS_DIR, "{0}.h5".format(id)), "w")
        data = f.create_dataset("samples",
                                np.shape(sample_array[:, :sum(runs[:(i+1)]),
                                                      :]))
        data[:, :] = sample_array[:, :sum(runs[:(i+1)]), :]
        f.close()

        # make various plots
        with h5py.File(os.path.join(RESULTS_DIR, "{0}.h5".format(id)),
                       "r") as f:
            samples = f["samples"][...]
        mcmc_result = make_plot(samples, x, y, yerr, id, RESULTS_DIR,
                                traces=True, tri=True, prediction=True)
    return mcmc_result


if __name__ == "__main__":
    id = 211000411
    x, y = load_k2_data(id, DATA_DIR)
    yerr = np.ones_like(y) * 1e-5
    p_init, p_err = calc_p_init(x, y, yerr)
    sub = 10
    xb, yb, yerrb = x[::sub], y[::sub], yerr[::sub]
    fit(xb, yb, yerrb, p_init)
