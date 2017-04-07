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


def butter_bandpass(freq, fs, order=5):
    nyq = 0.5 * fs
    cut = freq / nyq
    b, a = sps.butter(order, cut, btype='high')
    return b, a


def butter_bandpass_filter(data, freq, fs, order=5):
    b, a = butter_bandpass(freq, fs, order=order)
    y = sps.lfilter(b, a, data)
    return y


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

        fs = 1e-6
        period = 50*24*3600  # 50 days
        freq = 1./period

        plt.figure(1)
        plt.clf()
        for order in [3, 6, 9]:
            b, a = butter_bandpass(freq, fs, order=order)
            w, h = sps.freqz(b, a, worN=2000)
            plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

        plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
                '--', label='sqrt(0.5)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig("bandpass")

        # Filter a noisy signal.
        nsamples = 500
        print(nsamples)
        t = np.linspace(0, 100*24*3600, 1000)
        x = .1 * np.sin(2 * np.pi * freq * np.sqrt(t))
        x += .01 * np.cos(2 * np.pi * 2*freq * t + 0.1)
        x += .02 * np.cos(2 * np.pi * .5*freq * t + .11)
        x += .03 * np.cos(2 * np.pi * 3*freq * t)

        plt.figure(2)
        plt.clf()
        plt.plot(t, x, label='Noisy signal')
        y = butter_bandpass_filter(x, freq, fs, order=6)
        # plt.plot(t, y, label='Filtered signal (%g Hz)' % freq)
        plt.xlabel('time (seconds)')
        # plt.hlines([-a, a], 0, T, linestyles='--')
        # plt.grid(True)
        # plt.axis('tight')
        plt.legend(loc='upper left')

        plt.savefig("butter_demo")

        # b, a = sps.butter(4, 1./50, 'high', analog=True)
        # w, h = sps.freqs(b, a)
        # plt.plot(w, 20 * np.log10(abs(h)))
        # plt.xscale('log')
        # plt.title('Butterworth filter frequency response')
        # plt.xlabel('Frequency [radians / second]')
        # plt.ylabel('Amplitude [dB]')
        # plt.margins(0, 0.1)
        # plt.grid(which='both', axis='both')
        # plt.axvline(100, color='green') # cutoff frequency
        # plt.savefig("butter")
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
