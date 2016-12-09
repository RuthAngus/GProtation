# This script contains general functions for calculating an initial period
# and running the MCMC.

# coding: utf-8
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import h5py

import emcee3
from gatspy.periodic import LombScargle

from GProtation import make_plot, MyModel
import simple_acf as sa

DATA_DIR = "data/"
RESULTS_DIR = "results/"


def mcmc_fit(x, y, yerr, p_init, p_max, id, RESULTS_DIR, truths, burnin=500,
             nwalkers=12, nruns=10, full_run=500, diff_threshold=.5,
             n_independent=1000):
    """
    Run the MCMC
    """

    try:
        print("Total number of points  = ", sum([len(i) for i in x]))
        print("Number of light curve sections = ", len(x))
    except TypeError:
        print("Total number of points  = ", len(x))

    theta_init = np.log([np.exp(-12), np.exp(7), np.exp(-1), np.exp(-17),
                         p_init])
    runs = np.zeros(nruns) + full_run
    ndim = len(theta_init)

    print("p_init = ", p_init, "days, log(p_init) = ", np.log(p_init),
          "p_max = ", p_max)
    args = (x, y, yerr, np.log(p_init), p_max)

    # Time the LHF call.
    start = time.time()
    mod = MyModel(x, y, yerr, np.log(p_init), p_max)
    print("lnlike = ", mod.lnlike_split(theta_init), "lnprior = ",
          mod.Glnprior(theta_init), "\n")
    end = time.time()
    tm = end - start
    print("1 lhf call takes ", tm, "seconds")
    print("burn in will take", tm * nwalkers * burnin, "s")
    print("each run will take", tm * nwalkers * runs[0]/60, "mins")
    print("total = ", (tm * nwalkers * np.sum(runs) + tm * nwalkers *
                       burnin)/60, "mins")

    # Run MCMC.
    mod = MyModel(x, y, yerr, np.log(p_init), p_max)
    model = emcee3.SimpleModel(mod.lnlike_split, mod.Glnprior)
    p0 = [theta_init + 1e-4 * np.random.rand(ndim) for i in range(nwalkers)]
    ensemble = emcee3.Ensemble(model, p0)
#     moves = emcee3.moves.KDEMove()
#     sampler = emcee3.Sampler(moves)
    sampler = emcee3.Sampler()

    print("burning in...")
    total_start = time.time()
    ensemble = sampler.run(ensemble, burnin)

    flat = sampler.get_coords(flat=True)
    logprob = sampler.get_log_probability(flat=True)
    ensemble = emcee3.Ensemble(model, p0)

    # repeating MCMC runs.
    autocorr_times, mean_ind, mean_diff = [], [], []
    sample_array = np.zeros((nwalkers, sum(runs), ndim))
    for i, run in enumerate(runs):
        print("run {0} of {1}".format(i, len(runs)))
        print("production run, {0} steps".format(int(run)))
        start = time.time()
        ensemble = sampler.run(ensemble, run)
        end = time.time()
        print("time taken = ", (end - start)/60, "minutes")

        f = h5py.File(os.path.join(RESULTS_DIR, "{0}.h5".format(id)), "w")
        data = f.create_dataset("samples",
                                np.shape(sampler.get_coords(flat=True)))
        data[:, :] = sampler.get_coords(flat=True)
        f.close()

        print("samples = ", np.shape(sampler.get_coords(flat=True)))
        results = make_plot(sampler, x, y, yerr, id, RESULTS_DIR, truths,
                            traces=True, tri=True, prediction=True)
        nsteps, _ = np.shape(sampler.get_coords(flat=True))
        conv, autocorr_times, ind_samp, diff = \
                evaluate_convergence(sampler.get_coords(flat=True),
                                     autocorr_times, diff_threshold,
                                     n_independent)
        mean_ind.append(ind_samp)
        mean_diff.append(diff)
        print("Converged?", conv)
        if conv:
            break

    total_end = time.time()
    total_time = total_end - total_start
    print("Total time taken = ", total_time/60., "minutes", total_time/3600.,
          "hours")

    with open(os.path.join(RESULTS_DIR, "{0}_time.txt".format(id)), "w") as f:
        f.write("{}".format(total_time))

#     col = "b"
#     if conv:
#         col = "r"
#     if autocorr_times:
#         plt.clf()
#         plt.plot(autocorr_times, color=col)
#         plt.savefig(os.path.join(RESULTS_DIR, "{0}_acorr".format(id)))
#         plt.clf()
#         plt.plot(mean_ind, color=col)
#         plt.savefig(os.path.join(RESULTS_DIR, "{0}_ind".format(id)))
#         plt.clf()
#         plt.plot(mean_diff, color=col)
#         plt.savefig(os.path.join(RESULTS_DIR, "{0}_diff".format(id)))
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
    	A list of lists of autocorrelation times (averaged over
        chains/dimensions).
    mean_ind: (list)
        The number of independent samples (averaged over chains/dimensions).
    mean_diff: (list)
        The difference between this autocorrelation time and the last
	autocorrelation time.
    """
    converged = False
    try:
        acorr_t = emcee3.autocorr.integrated_time(flat, c=1)
    except emcee3.autocorr.AutocorrError:
        return converged, [], [], []
    mean_acorr.append(np.mean(acorr_t))
    mean_ind = len(flat) / np.mean(acorr_t)
    mean_diff = None
    if len(mean_acorr) > 1:
        mean_diff = np.mean(mean_acorr[-1] - mean_acorr[-2])
        if np.abs(mean_diff) < diff_threshold and mean_ind > 1000:
            converged = True
    return converged, mean_acorr, mean_ind, mean_diff


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
