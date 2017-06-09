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
from scipy.optimize import curve_fit

def calc_phase_and_amp(x, y, f):
    AT = np.vstack((x, np.ones((3, len(y)))))
    ATA = np.dot(AT, AT.T)

    arg = 2*np.pi*f*x
    AT[-2, :] = np.sin(arg)
    AT[-1, :] = np.cos(arg)

    # AT.shape = (153, nt)
    # shape: (151, nt) * (nt, 2) -> (151, 2)
    v = np.dot(AT[:-2, :], AT[-2:, :].T)
    ATA[:-2, -2:] = v
    ATA[-2:, :-2] = v.T

    # AT[-2:, :].shape = (2, nt)
    # (2, nt), (nt, 2)
    ATA[-2:, -2:] = np.dot(AT[-2:, :], AT[-2:, :].T)
    w = np.linalg.solve(ATA, np.dot(AT, y))

    A, B = w[-1], w[-2]
    phase = np.arctan(A/B)
    Amp = (np.abs(A) + np.abs(B))**.5
    # print("phase = ", phase/np.pi, "Amp = ", Amp**2)
    return phase, Amp


def calc_pgram_uncertainty(x, y, freq):

    # Fit for phase and amplitude.
    phase, A = calc_phase_and_amp(x, y, freq)

    # Remove signal from data.
    y_noise = y - A**2*np.sin(2*np.pi*freq*x + phase)

    # Calculate variance, sigma_n
    sigma_n = np.var(y_noise)

    N, T = len(x), x[-1] - x[0]
    return 3 * np.pi * sigma_n / (2 * N**.5 * T * A)


def calc_p_init(x, y, yerr, id, RESULTS_DIR="pgram_filtered_results_35",
                clobber=True):

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

        # Interpolate across gaps
        gap_days = 0.02043365
        time = np.arange(x[0], x[-1], gap_days)
        lin_interp = np.interp(time, x, y)
        x, y = time, lin_interp
        yerr = np.ones(len(y)) * 1e-5

        print("Calculating periodogram")
        ps = np.arange(.1, 100, .1)
        model = LombScargle().fit(x, y, yerr)
        pgram = model.periodogram(ps)
        plt.clf()
        plt.plot(ps, pgram)

        period = 35.  # days
        fs = 1./(x[1] - x[0])
        lowcut = 1./period
        # pgram = model.periodogram(ps)
        yfilt = butter_bandpass_filter(y, lowcut, fs, order=3, plot=False)

        # plt.clf()
        # plt.plot(x, y, label='Noisy signal')
        # plt.plot(x, yfilt, label='{0} day^(-1), {1} days'.format(lowcut,
        #                                                          period))
        # plt.xlabel('time (seconds)')
        # plt.grid(True)
        # plt.axis('tight')
        # plt.legend(loc='upper left')
        # plt.savefig("butter_filtered")

        # normalise data and variance.
        med = np.median(y)
        y -= med
        var = np.std(y)
        print("var = ", var)
        y /= var

        print("Calculating periodogram")
        ps = np.arange(.1, 100, .1)
        model = LombScargle().fit(x, yfilt, yerr)
        pgram = model.periodogram(ps)

        plt.plot(ps, pgram)
        plt.savefig(os.path.join(RESULTS_DIR, "{0}_pgram".format(id)))
        print("saving figure ", os.path.join(RESULTS_DIR,
                                             "{0}_pgram".format(id)))

        peaks = np.array([i for i in range(1, len(ps)-1) if pgram[i-1] <
                          pgram[i] and pgram[i+1] < pgram[i]])
        pgram_period = ps[pgram == max(pgram[peaks])][0]

        # Calculate the uncertainty.
        _freq = 1./pgram_period
        pgram_freq_err = calc_pgram_uncertainty(x, y, _freq)
        print(1./_freq, "period")
        print(_freq, "freq")
        print(pgram_freq_err, "pgram_freq_err")
        frac_err = pgram_freq_err/_freq
        print(frac_err, "frac_err")
        pgram_period_err = pgram_period * frac_err
        print(pgram_period_err, "pgram_period_err")
        print("pgram period = ", pgram_period, "+/-", pgram_period_err,
              "days")

        df = pd.DataFrame({"N": [id], "acf_period": [acf_period],
                           "acf_period_err": [err],
                           "pgram_period": [pgram_period],
                           "pgram_period_err": [pgram_period_err]})
        df.to_csv(fname)

    return acf_period, err, pgram_period, pgram_period_err


if __name__ == "__main__":

    # x = np.arange(0, 100, .1)
    # y = 3 * np.sin(2 * np.pi * (1./10) * x + np.pi/2)
    # y += np.random.randn(len(y)) * .0000000001
    # err = calc_pgram_uncertainty(x, y, 1./10)
    # print(err)
    # assert 0

    DIR = "/Users/ruthangus/projects/GProtation/code/kepler_diffrot_full/"
    truths = pd.read_csv(os.path.join(DIR, "par/final_table.txt"),
                         delimiter=" ")
    m = (truths.DELTA_OMEGA.values == 0)
    truths = truths.iloc[m]
    for i, id in enumerate(truths.N.values):
        x, y = load_suzanne_lcs(id, os.path.join(DIR, "final"))
        yerr = np.ones(len(y)) * 1e-5
        _, _, pgram, _ = calc_p_init(x, y, yerr, id, clobber=True)
