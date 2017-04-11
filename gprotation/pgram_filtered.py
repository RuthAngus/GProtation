"""
This script contains general functions for calculating an initial period
and running the MCMC.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import h5py
import scipy.signal as sps

from gatspy.periodic import LombScargle
from recover_suzannes import load_suzanne_lcs
import simple_acf as sa
from filtering import butter_bandpass, butter_bandpass_filter


def calc_p_init(x, y, yerr, id, RESULTS_DIR="pgram_results", clobber=True):
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

        # Sample rate and desired cutoff frequencies (in Hz).
        fs = 5000.0
        lowcut = 500.0

        # Filter a noisy signal.
        # T = 0.05
        # nsamples = T * fs
        # t = np.linspace(0, T, nsamples, endpoint=False)
        # a = 0.02
        # f0 = 600.0
        # x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
        # x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
        # x += a * np.cos(2 * np.pi * f0 * t + .11)
        # x += 0.03 * np.cos(2 * np.pi * 2000 * t)

        x *= 24*3600
        fs = x[1] - x[0]
        print(fs)

        yfilt = butter_bandpass_filter(y, lowcut, fs, order=6)

        plt.clf()
        plt.plot(x, y, label='Noisy signal')
        plt.plot(x, yfilt, label='Filtered signal (%g Hz)' % lowcut)
        plt.xlabel('time (seconds)')
        # plt.hlines([-a, a], 0, T, linestyles='--')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc='upper left')
        plt.savefig("butter_filtered")
        assert 0


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


if __name__ == "__main__":
    DIR = "/Users/ruthangus/projects/GProtation/code/kepler_diffrot_full/"
    truths = pd.read_csv(os.path.join(DIR, "par/final_table.txt"),
                         delimiter=" ")
    m = (truths.DELTA_OMEGA.values == 0)
    truths = truths.iloc[m]
    for i, id in enumerate(truths.N.values):
        x, y = load_suzanne_lcs(id, os.path.join(DIR, "final"))
        yerr = np.ones(len(y)) * 1e-5
        calc_p_init(x, y, yerr, id, clobber=True)
