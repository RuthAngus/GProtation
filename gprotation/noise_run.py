# measure rotation periods for white noise.
from __future__ import print_function
import numpy as np
import os
import gprot_fit as gp
import matplotlib.pyplot as plt
import gprot_fit as gp

def noise_fit(id, RESULTS_DIR):

    gap_days = .02043365
    x = np.arange(0, 200, gap_days)
    y = np.random.randn(len(x)) * 1e-5
    yerr = np.ones_like(y) * 1e-5

    fit = gp.fit(x, y, yerr, id, RESULTS_DIR, lc_size=201)
    fit.gp_fit(burnin=1000, nwalkers=16, nruns=5, full_run=1000, nsets=1)


if __name__ == "__main__":
    id = 0
    RESULTS_DIR = "noise"
    noise_fit(id, RESULTS_DIR)
