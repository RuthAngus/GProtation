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

    RESULTS_DIR = "results_periodic_gp"
    DATA_DIR = "periodic_gp_simulations"
#     RESULTS_DIR = "results_aperiodic_gp"
#     DATA_DIR = "aperiodic_gp_simulations"

    truths = pd.read_csv("gp_truths.csv")

    sid = str(int(i)).zfill(4)
    print(sid, "of", 45)
    x, y = np.genfromtxt(os.path.join(DATA_DIR, "{0}.txt".format(sid))).T
    yerr = np.ones_like(y) * 1e-5
    x_n, y_n = np.genfromtxt(os.path.join(DATA_DIR,
                             "{0}_noise.txt".format(sid))).T
    yerr_n = np.ones_like(y_n) * 1e-5
    plt.clf()
    plt.plot(x, y, "k.")
    plt.savefig("test")

    # sigma clip
    x, y, yerr = sigma_clip(x, y, yerr, 5)
    x_n, y_n, yerr_n = sigma_clip(x_n, y_n, yerr_n, 5)

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

    if p_init < .5:
        p_init = 20.
    if p_init_n < .5:
        p_init_n = 20.
    burnin, nwalkers, nruns, full_run = 500, 12, 10, 500
    print("p_init = ", p_init, "p_init_n = ", p_init_n)

    # fast settings:
#     burnin, nwalkers, nruns, full_run = 2, 12, 2, 50
#     x, y, yerr = x[::100], y[::100], yerr[::100]
#     x_n, y_n, yerr_n = x_n[::100], y_n[::100], yerr_n[::100]

    # set prior bounds
    plims = np.log([.5*p_init, 1.5*p_init])
    plims_n = np.log([.5*p_init_n, 1.5*p_init_n])

    # run on noisy and noise free
    trths_p = [truths.lnA.values[i], truths.lnl_p.values[i],
             truths.lngamma.values[i], truths.lnsigma.values[i],
             truths.lnperiod.values[i]]
    trths_a = [truths.lnA.values[i], truths.lnl_p.values[i], None,
  	       truths.lnsigma.values[i], None]
    mcmc_fit(x, y, yerr, p_init, plims, sid, RESULTS_DIR, trths_p,
	     burnin=burnin, nwalkers=nwalkers, nruns=nruns, full_run=full_run)
    mcmc_fit(x_n, y_n, yerr_n, p_init_n, plims_n, "{0}_n".format(sid), trths_p,
             RESULTS_DIR, burnin=burnin, nwalkers=nwalkers, nruns=nruns,
             full_run=full_run)

if __name__ == "__main__":

    pool = Pool()
    results = pool.map(recover, range(45))

#     for i in range(29):
# 	    recover(i)
