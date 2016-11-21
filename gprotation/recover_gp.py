from __future__ import print_function
import numpy as np
from GProt import calc_p_init, mcmc_fit
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py
from Kepler_ACF import corr_run
from gatspy.periodic import LombScargle
from recover_suzannes import make_lists, make_gaps

def load_suzanne_lcs(id):
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
    sid = str(int(i)).zfill(4)

    truths = pd.read_csv("gp_truths.csv")

    # select either periodic or non-periodic
    RESULTS_DIR = "results_periodic_gp"
#     RESULTS_DIR = "results_aperiodic_gp"

    DATA_DIR = "periodic_gp_simulations"
    trths = [truths.lnA.values[i], truths.lnl_p.values[i],
             truths.lngamma.values[i], truths.lnsigma.values[i],
             truths.lnperiod.values[i]]
    if RESULTS_DIR == "results_aperiodic_gp":
        DATA_DIR = "aperiodic_gp_simulations"
        trths = [truths.lnA.values[i], truths.lnl_p.values[i], None,
                   truths.lnsigma.values[i], None]

    sid = str(int(i)).zfill(4)
    print(sid, "of", 333)
    x, y = np.genfromtxt(os.path.join(DATA_DIR, "{0}.txt".format(sid))).T
    yerr = np.ones_like(y) * 1e-5
    x_n, y_n = np.genfromtxt(os.path.join(DATA_DIR,
                             "{0}_noise.txt".format(sid))).T
    yerr_n = np.ones_like(y_n) * 1e-5

    # sigma clip
    x, y, yerr = sigma_clip(x, y, yerr, 5)
    x_n, y_n, yerr_n = sigma_clip(x_n, y_n, yerr_n, 5)

    ppd = 4  # cut off at 200 days, 4 points per day
    xb, yb, yerrb = make_gaps(x, y, yerr, ppd)
    xb_n, yb_n, yerrb_n = make_gaps(x_n, y_n, yerr_n, ppd)

    # make data into a list of lists, 200 days each
    xb, yb, yerrb = make_lists(xb, yb, yerrb, 200)
    xb_n, yb_n, yerrb_n = make_lists(xb_n, yb_n, yerrb_n, 200)

    # find p_init
    acf_period_n, a_err_n, pgram_period_n, p_err_n = \
            calc_p_init(x_n, y_n, yerr_n, "{0}_n".format(sid), RESULTS_DIR,
                        clobber=True)
    acf_period, a_err, pgram_period, p_err = calc_p_init(x, y, yerr, sid,
                                                         RESULTS_DIR,
                                                         clobber=True)

    # set initial period
    p_init = acf_period
    p_init_n = acf_period_n

    p_max = np.log((xb[0][-1] - xb[0][0]) / 2.)
    if p_init > np.exp(p_max):
        p_init = 40
    elif p_init < .5:
        p_init = 10
    burnin, nwalkers, nruns, full_run = 1000, 16, 5, 1000

    assert p_init < np.exp(p_max), "p_init > p_max"

    # fast settings:
#     burnin, nwalkers, nruns, full_run = 2, 12, 5, 50
#     xb[0], yb[0], yerrb[0] = xb[0][::100], yb[0][::100], yerrb[0][::100]
#     xb_n[0], yb_n[0], yerrb_n[0] = xb_n[0][::100], yb_n[0][::100], \
#         yerrb_n[0][::100]

    # run on noisy and noise free
#     mcmc_fit(xb[0], yb[0], yerrb[0], p_init, p_max, sid, RESULTS_DIR,
    mcmc_fit(xb[:2], yb[:2], yerrb[:2], p_init, p_max, sid, RESULTS_DIR,
             truths=trths, burnin=burnin, nwalkers=nwalkers, nruns=nruns,
             full_run=full_run, diff_threshold=.5, n_independent=1000)
#     mcmc_fit(xb_n[0], yb_n[0], yerrb_n[0], p_init_n, p_max,
    mcmc_fit(xb_n[:2], yb_n[:2], yerrb_n[:2], p_init_n, p_max,
             "{0}_n".format(sid), RESULTS_DIR, truths=trths, burnin=burnin,
             nwalkers=nwalkers, nruns=nruns, full_run=full_run,
             diff_threshold=.5, n_independent=1000)

if __name__ == "__main__":

    pool = Pool()
    results = pool.map(recover, range(333))

#     recover(0)

#     for i in range(333):
# 	    recover(i)
