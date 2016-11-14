# This script contains general functions for calculating an initial period
# and running the MCMC.

# coding: utf-8
from __future__ import print_function
import numpy as np
from GProtation import make_plot, lnprob, Glnprob
from Kepler_ACF import corr_run
import h5py
from gatspy.periodic import LombScargle
import os
import time
import emcee
import pyfits
import matplotlib.pyplot as plt
import pandas as pd

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


def calc_p_init(x, y, yerr, id, RESULTS_DIR, clobber=False):
    fname = os.path.join(RESULTS_DIR, "{0}_acf_pgram_results.txt".format(id))
    if not clobber and os.path.exists(fname):
        print("Previous ACF pgram result found")
        df = pd.read_csv(fname)
        m = df.N.values == id
        acf_period = df.acf_period.values[m]
        err = df.acf_period_err.values[m]
        pgram_period = df.pgram_period.values[m]
        pgram_period_err = df.pgram_period_err.values[m]
    else:
        print("Calculating ACF")
        acf_period, err, lags, acf = corr_run(x, y, yerr, id, RESULTS_DIR)

        print("Calculating periodogram")
        ps = np.arange(.1, 100, .1)
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

        df = pd.DataFrame({"N": [id], "acf_period": [acf_period],
                           "acf_period_err": [err],
                           "pgram_period": [pgram_period],
                           "pgram_period_err": [pgram_period_err]})
        df.to_csv(fname)
    return acf_period, err, pgram_period, pgram_period_err


def mcmc_fit(x, y, yerr, p_init, plims, id, RESULTS_DIR, truths, burnin=500,
             nwalkers=12, nruns=10, full_run=500, parallel=False):
    """
    Run the MCMC
    """

    print("total number of points = ", len(x))
    theta_init = np.log([np.exp(-12), np.exp(7), np.exp(-1), np.exp(-17),
                         p_init])
    runs = np.zeros(nruns) + full_run
    ndim = len(theta_init)
    inits = [1, 1, 1, 1, np.log(.5*p_init)]
    p0 = [theta_init + inits * np.random.rand(ndim) for i in range(nwalkers)]

    p_max = np.log((x[-1] - x[0]) / 2.)

    # comment this line for Tim's initialisation
    p0 = [theta_init + 1e-4 * np.random.rand(ndim) for i in range(nwalkers)]

    print("p_init = ", p_init, "days, log(p_init) = ", np.log(p_init),
          "p_max = ", p_max)
    args = (x, y, yerr, np.log(p_init), p_max)

    # Time the LHF call.
    start = time.time()
    print("lnprob = ", Glnprob(theta_init, x, y, yerr, np.log(p_init),
                               p_max)[0], "\n")
    end = time.time()
    tm = end - start
    print("1 lhf call takes ", tm, "seconds")
    print("burn in will take", tm * nwalkers * burnin, "s")
    print("each run will take", tm * nwalkers * runs[0]/60, "mins")
    print("total = ", (tm * nwalkers * np.sum(runs) + tm * nwalkers *
                       burnin)/60, "mins")


    # Run MCMC.
    if parallel:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, Glnprob, args=args,
                                        threads=15)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, Glnprob, args=args)
    print("burning in...")
    p0, _, state, prob = sampler.run_mcmc(p0, burnin)

    flat = np.reshape(sampler.chain, (nwalkers * burnin, ndim))
    logprob = flat[:, -1]
    ml = logprob == max(logprob)
    theta_init = flat[np.where(ml)[0][0], :]  # maximum likelihood sample

    # uncomment this line for Tim's initialisation
#     p0 = [theta_init + 1e-4 * np.random.rand(ndim) for i in range(nwalkers)]

    sample_array = np.zeros((nwalkers, sum(runs), ndim + 1))  # +1 for blobs
    for i, run in enumerate(runs):
        print("run {0} of {1}".format(i, len(runs)))
        sampler.reset()
        print("production run, {0} steps".format(int(run)))
        start = time.time()
        p0, _, state, prob = sampler.run_mcmc(p0, run)
        end = time.time()
        print("time taken = ", (end - start)/60, "minutes")

        # save samples
        sample_array[:, sum(runs[:i]):sum(runs[:(i+1)]), :-1] = \
            np.array(sampler.chain)
        sample_array[:, sum(runs[:i]):sum(runs[:(i+1)]), -1] = \
                np.array(sampler.blobs).T
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
        results = make_plot(samples, x, y, yerr, id, RESULTS_DIR, truths, traces=True,
                            tri=True, prediction=True)
        return samples, results

if __name__ == "__main__":
    id = 211000411
    x, y = load_k2_data(id, DATA_DIR)
    yerr = np.ones_like(y) * 1e-5
    p_init, p_err = calc_p_init(x, y, yerr)
    sub = 10
    xb, yb, yerrb = x[::sub], y[::sub], yerr[::sub]
    fit(xb, yb, yerrb, p_init)
