from __future__ import print_function
import numpy as np
from GProt import calc_p_init, mcmc_fit
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py
import math


def make_lists(xb, yb, yerrb, l):
    nlists = int(math.ceil((xb[-1] - xb[0]) / l))
    xlist, ylist, yerrlist= [], [], []
    masks = np.arange(nlists + 1) * l
    for i in range(nlists):
        m = (masks[i] < xb) * (xb < masks[i+1])
        xlist.append(xb[m])
        ylist.append(yb[m])
        yerrlist.append(yerrb[m])
    return xlist, ylist, yerrlist


def make_gaps(x, y, yerr, points_per_day):
    nkeep = points_per_day * (x[-1] - x[0])
    m = np.zeros(len(x), dtype=bool)
    l = np.random.choice(np.arange(len(x)), nkeep)
    for i in l:
        m[i] = True
    inds = np.argsort(x[m])
    return x[m][inds], y[m][inds], yerr[m][inds]


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

#     DATA_DIR = "../code/simulations/kepler_diffrot_full/noise_free"
#     RESULTS_DIR = "results_nf"

    RESULTS_DIR = "results_sigma"  # just 2 sets of 200 days
#     RESULTS_DIR = "results"
    DATA_DIR = "../code/simulations/kepler_diffrot_full/final"

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0

    id = truths.N.values[m][i]
    sid = str(int(id)).zfill(4)
    print(id, i, "of", len(truths.N.values[m]))
    x, y = load_suzanne_lcs(sid, DATA_DIR)
    yerr = np.ones_like(y) * 1e-5
    if RESULTS_DIR == "results_nf":
        yerr *= 1e-15

    # sigma clip
    x, y, yerr = sigma_clip(x, y, yerr, 5)

    # calculate the variance
    var = np.var(y)
    burnin, nwalkers, nruns, full_run = 1000, 16, 20, 500
    if np.log(var) < -13:
        burnin, nwalkers, nruns, full_run = 1000, 16, 20, 1000

    ppd = 4  # cut off at 200 days, 4 points per day
    xb, yb, yerrb = make_gaps(x, y, yerr, ppd)

    # make data into a list of lists, 200 days each
    xb, yb, yerrb = make_lists(xb, yb, yerrb, 200)

    # find p_init
    acf_period, a_err, pgram_period, p_err = calc_p_init(x, y, yerr, sid,
                                                         RESULTS_DIR,
                                                         clobber=False)

    # set initial period
    p_init = acf_period
    p_max = np.log((xb[0][-1] - xb[0][0]) / 2.)
    if p_init > np.exp(p_max):
        p_init = 40
    elif p_init < .5:
        p_init = 10
    burnin, nwalkers, nruns, full_run = 1000, 16, 5, 1000

    assert p_init < np.exp(p_max), "p_init > p_max"

    # fast settings
#     burnin, nwalkers, nruns, full_run = 2, 12, 5, 50
#     xb[0], yb[0], yerrb[0] = xb[0][::1000], yb[0][::1000], yerrb[0][::1000]

    trths = [None, None, None, None, np.log(truths.P_MIN.values[m][i])]
#     mcmc_fit(xb, yb, yerrb, p_init, p_max, sid, RESULTS_DIR,
#     mcmc_fit(xb[0], yb[0], yerrb[0], p_init, p_max, sid, RESULTS_DIR,
    mcmc_fit(xb[:2], yb[:2], yerrb[:2], p_init, p_max, sid, RESULTS_DIR,
             truths=trths, burnin=burnin, nwalkers=nwalkers, nruns=nruns,
             full_run=full_run, diff_threshold=.5, n_independent=1000)

if __name__ == "__main__":

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0

    pool = Pool()
    results = pool.map(recover, range(len(truths.N.values[m][:100])))

#     recover(2)

#     for i in range(len(truths.N.values[m])):
# 	    recover(i)
