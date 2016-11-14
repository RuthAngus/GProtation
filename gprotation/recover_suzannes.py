from __future__ import print_function
import numpy as np
from GProt import calc_p_init, mcmc_fit
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py


def load_suzanne_lcs(id, DATA_DIR):
    sid = str(int(id)).zfill(4)
    x, y = np.genfromtxt(os.path.join(DATA_DIR,
                                      "lightcurve_{0}.txt".format(sid))).T
    return x - x[0], y - 1


def sigma_clip(x, y, yerr, nsigma):
    med = np.median(y)
    std = (sum((med - y)**2)/float(len(y)))**.5
    m = np.abs(y - med) > (nsigma * std)
    return x[~m], y[~m], yerr[~m]

def recover(i):

#     RESULTS_DIR = "results"
#     RESULTS_DIR = "results_prior"
#     RESULTS_DIR = "results_Gprior"
#     RESULTS_DIR = "results_initialisation"

    RESULTS_DIR = "results_sigma"
    DATA_DIR = "../code/simulations/kepler_diffrot_full/final"

#     DATA_DIR = "../code/simulations/kepler_diffrot_full/noise_free"
#     RESULTS_DIR = "results_nf"

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0

    id = truths.N.values[m][i]
    sid = str(int(id)).zfill(4)
    print(id, i, "of", len(truths.N.values[m]))
    x, y = load_suzanne_lcs(sid, DATA_DIR)
    yerr = np.ones_like(y) * 1e-5

    # sigma clip
    x, y, yerr = sigma_clip(x, y, yerr, 5)

    # calculate the variance
    var = np.var(y)
    burnin, nwalkers, nruns, full_run = 1000, 16, 20, 500
    if np.log(var) < -13:
        burnin, nwalkers, nruns, full_run = 1000, 16, 20, 1000

    c, sub = 200, 10  # cut off at 200 days
    mc = x < c
    xb, yb, yerrb = x[mc][::sub], y[mc][::sub], yerr[mc][::sub]

    # find p_init
    acf_period, a_err, pgram_period, p_err = calc_p_init(x, y, yerr, sid,
                                                         RESULTS_DIR,
                                                         clobber=False)

    # set initial period
    p_init = acf_period
    p_max = np.log((x[-1] - x[0]) / 2.)
    if p_init > np.exp(p_max) or p_init < .5:
        p_init = 10
        burnin, nwalkers, nruns, full_run = 1000, 16, 20, 1000
    if p_init > 40:
        burnin, nwalkers, nruns, full_run = 1000, 16, 20, 1000
    burnin, nwalkers, nruns, full_run = 5000, 16, 20, 1000

    assert p_init < np.exp(p_max), "p_init > p_max"

    # set prior bounds
    plims = np.log([.5*p_init, 1.5*p_init])
#     plims = np.log([.1*p_init, 5*p_init])

    # fast settings
#     burnin, nwalkers, nruns, full_run = 2, 12, 2, 50
#     xb, yb, yerrb = xb[::10], yb[::10], yerrb[::10]

    trths = [None, None, None, None, truths.P_MIN.values[m][i]]
    mcmc_fit(xb, yb, yerrb, p_init, plims, sid, RESULTS_DIR, truths=trths,
	     burnin=burnin, nwalkers=nwalkers, nruns=nruns, full_run=full_run)

if __name__ == "__main__":

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0

#     recover(2)

    pool = Pool()
    results = pool.map(recover, range(len(truths.N.values[m][:100])))

#     for i in range(len(truths.N.values[m])):
#     for i in range(10):
# 	    recover(i)
