from __future__ import print_function
import numpy as np
from GProt import calc_p_init, mcmc_fit
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py
import math
import gprot_fit as gp


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


def gp_fit(x, y, yerr, sid, RESULTS_DIR):

    # sigma clip
    x, y, yerr = sigma_clip(x, y, yerr, 5)

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

    assert p_init < np.exp(p_max), "p_init > p_max"

    l = truths.N.values == int(sid)
    true_period = truths.P_MIN.values[l]
    trths = [None, None, None, None, np.log(true_period)]
    mcmc_fit(xb[:2], yb[:2], yerrb[:2], p_init, p_max, sid, RESULTS_DIR,
             truths=trths, burnin=burnin, nwalkers=nwalkers, nruns=nruns,
             full_run=full_run, diff_threshold=.5, n_independent=1000)


def recover(i):

    RESULTS_DIR = "acf_results"
    DATA_DIR = "kepler_diffrot_full/final"

    DIR = "kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0

    id = truths.N.values[m][i]
    sid = str(int(id)).zfill(4)
    print(id, i, "of", len(truths.N.values[m]))
    x, y = load_suzanne_lcs(sid, DATA_DIR)
    yerr = np.ones_like(y) * 1e-5

    fit = gp.fit(x, y, yerr, sid, RESULTS_DIR)
    # fit.gp_fit(burnin=1000, nwalkers=20, nruns=10, full_run=1000, nsets=2)


if __name__ == "__main__":

    DIR = "kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0

    for i in range(len(truths.N.values[m])):
        recover(i)
    # pool = Pool()
    # results = pool.map(recover, range(len(truths.N.values[m])))
