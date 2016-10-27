from __future__ import print_function
import numpy as np
from GProt import calc_p_init, mcmc_fit
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py

DATA_DIR = "../code/simulations/kepler_diffrot_full/final"
RESULTS_DIR = "results/"


def load_suzanne_lcs(id):
    sid = str(int(id)).zfill(4)
    x, y = np.genfromtxt(os.path.join(DATA_DIR,
                                      "lightcurve_{0}.txt".format(sid))).T
    return x - x[0], y - 1


def make_new_df(truths, RESULTS_DIR):
    m = truths.DELTA_OMEGA.values == 0
    mcmc, acf_pgram = [pd.DataFrame() for i in range(2)]
    for i, id in enumerate(truths.N.values[m]):
        sid = str(int(id)).format(4)
        mcmc_fname = os.path.join(RESULTS_DIR,
                                  "{0}_mcmc_result.csv".format(sid))
        acf_pgram_fname = os.path.join(RESULTS_DIR,
                                       "{0}_acf_pgram_result.csv".format(sid))
        if os.path.exists(mcmc_fname):

            mcmc.append(pd.read_csv(mcmc_fname))
            acf_pgram.append(pd.read_csv(acf_pgram_fname))
    truths_s = pd.merge(truths[m], mcmc, acf_pgram, on="N")
    truths_s.to_csv("truths_extended.csv")
    return truths_s

def load_samples(id):
    fname = os.path.join(RESULTS_DIR, "{0}.h5".format(str(int(id)).zfill(4)))
    if os.path.exists(fname):
        with h5py.File(fname, "r") as f:
            samples = f["samples"][...]
    nwalkers, nsteps, ndims = np.shape(samples)
    return np.reshape(samples[:, :, 4], nwalkers * nsteps)

def sigma_clip(x, y, yerr, nsigma):
    med = np.median(y)
    std = (sum((med - y)**2)/float(len(y)))**.5
    m = np.abs(y - med) > (nsigma * std)
    return x[~m], y[~m], yerr[~m]

def recover(i):
    sid = str(int(i)).zfill(4)

#     RESULTS_DIR = "results"
    RESULTS_DIR = "results_prior"

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0

    id = truths.N.values[m][i]
    print(id, i, "of", len(truths.N.values[m]))
    x, y = load_suzanne_lcs(sid)
    yerr = np.ones_like(y) * 1e-5

    # sigma clip
    x, y, yerr = sigma_clip(x, y, yerr, 5)

    # calculate the variance
    var = np.var(y)
    burnin, nwalkers, nruns, full_run = 1000, 12, 10, 500
    if np.log(var) < -13:
        burnin, nwalkers, nruns, full_run = 1000, 16, 20, 500

    # find p_init
    acf_period, a_err, pgram_period, p_err = calc_p_init(x, y, yerr, sid,
                                                         RESULTS_DIR)
    # set initial period
    p_init = acf_period
    if p_init > 100 or p_init < 0:
        p_init = 10
    if p_init > 40:
        burnin, nwalkers, nruns, full_run = 1000, 16, 20, 500

    # set prior bounds
    plims = np.log([.5*p_init, 1.5*p_init])
#     plims = np.log([.1*p_init, 5*p_init])

    c, sub = 200, 10  # cut off at 200 days
    mc = x < c
    xb, yb, yerrb = x[mc][::sub], y[mc][::sub], yerr[mc][::sub]
    mcmc_fit(xb, yb, yerrb, p_init, plims, sid, RESULTS_DIR,
	     burnin=burnin, nwalkers=nwalkers, nruns=nruns, full_run=full_run)

if __name__ == "__main__":

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0
    pool = Pool()
    results = pool.map(recover, range(len(truths.N.values[m])))

#     for i in range(len(truths.N.values[m])):
# 	    recover(i)
