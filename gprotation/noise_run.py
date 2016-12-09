# measure rotation periods for white noise.
from __future__ import print_function
import numpy as np
import os
import gprot_fit as gp
import matplotlib.pyplot as plt
import gprot_fit as gp
import emcee2_recover_suzannes as e2

def noise_fit(id, RESULTS_DIR, emcee2=False):

    gap_days = .02043365
    x = np.arange(0, 200, gap_days)
    y = np.random.randn(len(x)) * 1e-5
    yerr = np.ones_like(y) * 1e-5

    if emcee2:
        e2.gp_fit(x, y, yerr, "{}2".format(id), RESULTS_DIR)
    else:  # emcee 3
        fit = gp.fit(x, y, yerr, id, RESULTS_DIR)
        fit.gp_fit(burnin=5000, nwalkers=16, nruns=10, full_run=1000, nsets=1)


def noisy_sinusoid(period, amp, id, RESULTS_DIR, emcee2=False):

    gap_days = .02043365
    x = np.arange(0, 200, gap_days)
    y = amp*np.sin(2*np.pi*(1./period)*x)
    y += np.random.randn(len(x)) * 1e-5
    yerr = np.ones_like(y) * 1e-5

    if emcee2:
        e2.gp_fit(x, y, yerr, "{}2".format(id), RESULTS_DIR)
    fit = gp.fit(x, y, yerr, id, RESULTS_DIR)
    fit.gp_fit(burnin=3000, nwalkers=16, nruns=5, full_run=1000, nsets=1)
#     fit.gp_fit(burnin=2, nwalkers=12, nruns=2, full_run=50, nsets=1)


if __name__ == "__main__":
    RESULTS_DIR = "noise"
#     noise_fit(0, RESULTS_DIR)
    noise_fit(0, RESULTS_DIR, emcee2=True)
#     noisy_sinusoid(10, 1e-3, 1, RESULTS_DIR)
#     noisy_sinusoid(10, 1e-3, 1, RESULTS_DIR, emcee2=True)
