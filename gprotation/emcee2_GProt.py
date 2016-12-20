# This script contains general functions for calculating an initial period
# and running the MCMC.

# coding: utf-8
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import pyfits

import os
import time

import emcee
from gatspy.periodic import LombScargle

from emcee2_GProtation import make_plot, Glnprob_split
import simple_acf as sa

DATA_DIR = "data/"
RESULTS_DIR = "results/"


def mcmc_fit(x, y, yerr, p_init, p_max, id, RESULTS_DIR, truths, burnin=500,
             nwalkers=12, nruns=10, full_run=500, diff_threshold=.5,
             n_independent=1000, parallel=False):
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

    # comment this line for Tim's initialisation
    p0 = [theta_init + 1e-4 * np.random.rand(ndim) for i in range(nwalkers)]

    print("p_init = ", p_init, "days, log(p_init) = ", np.log(p_init),
          "p_max = ", p_max)
    args = (x, y, yerr, np.log(p_init), p_max)

    # Time the LHF call.
    start = time.time()
    print("lnprob = ", Glnprob_split(theta_init, x, y, yerr, np.log(p_init),
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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, Glnprob_split,
                                        args=args, threads=15)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, Glnprob_split,
                                        args=args)
    print("burning in...")
    p0, _, state, prob = sampler.run_mcmc(p0, burnin)

    flat = np.reshape(sampler.chain, (nwalkers * burnin, ndim))
    logprob = flat[:, -1]
    ml = logprob == max(logprob)
    theta_init = flat[np.where(ml)[0][0], :]  # maximum likelihood sample

    # uncomment this line for Tim's initialisation
#     p0 = [theta_init + 1e-4 * np.random.rand(ndim) for i in range(nwalkers)]

    # repeating MCMC runs.
    autocorr_times, mean_ind, mean_diff = [], [], []
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
        make_plot(samples, x, y, yerr, id, RESULTS_DIR, truths,
                  traces=True, tri=True, prediction=True)
        _, nsteps, _ = np.shape(samples)
        flat = np.reshape(samples[:, :, :5], (nwalkers*nsteps, ndim))
        conv, autocorr_times, ind_samp, diff = \
            evaluate_convergence(flat, autocorr_times, diff_threshold,
                                 n_independent)
        mean_ind.append(ind_samp)
        mean_diff.append(diff)
        print(conv)
        if conv:
            break

    col = "b"
    if conv:
        col = "r"
    plt.clf()
    plt.plot(autocorr_times, color=col)
    plt.savefig(os.path.join(RESULTS_DIR, "{0}_acorr".format(id)))
    plt.clf()
    plt.plot(mean_ind, color=col)
    plt.savefig(os.path.join(RESULTS_DIR, "{0}_ind".format(id)))
    plt.clf()
    plt.plot(mean_diff, color=col)
    plt.savefig(os.path.join(RESULTS_DIR, "{0}_diff".format(id)))
    return


def evaluate_convergence(flat, mean_acorr, diff_threshold=0,
                         n_independent=1000):
    """
    Calculates the autocorrelation time of flat and appends to mean_acorr.
    Also calculates the number of independent samples and the difference
    between adjacent autocorrelation times estimates.

    Params:
    ------
    flat: (ndarray)
    	flattened array of emcee samples
    mean_acorr: (list)
    	list of lists of autocorrelation times.
    diff_threshold: (int) default = 0
	The threshold reached by the autocorrelation time difference which
	constitues convergence.
    ind_threshold: (int) default = 1000
	The minimum number of independent samples
    Returns
    -------
    converged: (boolean)
    	If True then convergence criteria have been satisfied.
    mean_acorr: (list)
    	A list of lists of autocorrelation times (averaged over chains/dimensions).
    mean_ind: (list)
        The number of independent samples (averaged over chains/dimensions).
    mean_diff: (list)
        The difference between this autocorrelation time and the last
	autocorrelation time.
    """
    converged = False
    try:
        acorr_t = emcee.autocorr.integrated_time(flat, c=1)
    except:
        acorr_t = emcee.autocorr.integrated_time(flat)
    mean_acorr.append(np.mean(acorr_t))
    mean_ind = len(flat) / np.mean(acorr_t)
    mean_diff = None
    if len(mean_acorr) > 1:
        mean_diff = np.mean(mean_acorr[-1] - mean_acorr[-2])
        if np.abs(mean_diff) < diff_threshold and mean_ind > 1000:
            converged = True
    return converged, mean_acorr, mean_ind, mean_diff


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
        acf_period, acf, lags, rvar = sa.simple_acf(x, y)
        err = .1 * acf_period
        plt.clf()
        plt.plot(lags, acf)
        plt.axvline(acf_period, color="r")
        plt.xlabel("Lags (days)")
        plt.ylabel("ACF")
        plt.savefig(os.path.join(RESULTS_DIR, "{0}_acf".format(id)))
        print("saving figure ", os.path.join(RESULTS_DIR,
                                             "{0}_acf".format(id)))

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
