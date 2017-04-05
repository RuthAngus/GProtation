from __future__ import print_function
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def acf_plot(truths_e,
             DIR="/Users/ruthangus/projects/GProtation/documents/figures"):
    """
    Plot the acf results.
    """

    m = (truths_e.DELTA_OMEGA.values == 0)
            # * (truths_e.acf_period.values > 0)
    N = truths_e.N.values[m]
    true = truths_e.P_MIN.values[m]
    acfs = truths_e.acf_period.values[m]
    acf_errs = truths_e.acf_period_err.values[m]
    amp = truths_e.AMP.values[m]
    m = acfs == 0
    acfs[m] = np.zeros(len(acfs[m])) + 1e-1
    tlv = pd.read_csv("telaviv_acf_output.txt")
    m = (truths_e.DELTA_OMEGA.values == 0)
    m = truths_e.N.values[m]
    acfs = tlv.period.values[m]
    acf_errs = tlv.period_err.values[m]

    print(len(true), "stars")
    # acf plot
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(np.log(xs), np.log(xs), "k-", alpha=.3, lw=.8, zorder=0)
    plt.plot(np.log(xs), np.log(xs) - 2./3, "k--", alpha=.3, lw=.8, zorder=0)
    plt.plot(np.log(xs), np.log(xs) + 2./3, "k--", alpha=.3, lw=.8, zorder=0)

    plt.errorbar(np.log(true), np.log(acfs), yerr=(acf_errs/acfs), fmt="k.",
                 capsize=0, ecolor=".7", alpha=.4, ms=1, zorder=1,
                 elinewidth=.8)
    plt.scatter(np.log(true), np.log(acfs), c=np.log(amp), edgecolor=".5",
                cmap="GnBu_r", vmin=min(np.log(amp)), vmax=max(np.log(amp)),
                s=10, lw=.2, zorder=2)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.xlim(0, 4)
    plt.ylim(-2, 6)
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period~ACF~Method})$")
    plt.subplots_adjust(bottom=.14)
    plt.savefig(os.path.join(DIR, "compare_acf.pdf"))
    # plt.savefig("compare_acf.pdf"))

    plt.clf()
    plt.hist(np.log(acfs) - np.log(true), histtype="stepfilled", color="w")
    plt.xlabel("$\ln(\mathrm{ACF~Period}) - \ln(\mathrm{True~Period})$")
    plt.savefig("acf_hist.pdf")

    return MAD(np.log(true), np.log(acfs)), MAD(true, acfs), \
        MAD_rel(true, acfs), RMS(true, acfs)

def pgram_plot(truths_e,
               DIR="/Users/ruthangus/projects/GProtation/documents/figures"):
    """
    Plot the pgram results.
    """

    m = (truths_e.DELTA_OMEGA.values == 0)
            # * (truths_e.acf_period.values > 0)
    N = truths_e.N.values[m]
    true = truths_e.P_MIN.values[m]
    pgram = truths_e.pgram_period.values[m]
    amp = truths_e.AMP.values[m]

    print(len(true))
    # pgram plot
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(np.log(xs), np.log(xs), "k-", alpha=.3, zorder=0)
    plt.plot(np.log(xs), np.log(xs) - 2./3, "k--", alpha=.3, zorder=0)
    plt.plot(np.log(xs), np.log(xs) + 2./3, "k--", alpha=.3, zorder=0)
    plt.scatter(np.log(true), np.log(pgram), c=np.log(amp), edgecolor=".5",
                cmap="GnBu_r", vmin=min(np.log(amp)), vmax=max(np.log(amp)),
                s=20, zorder=1, lw=.2)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.xlim(0, 4)
    plt.ylim(-2, 6)
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period~LS~Periodogram~method})$")
    plt.savefig(os.path.join(DIR, "compare_pgram.pdf"))

    return MAD(np.log(true), np.log(pgram)), MAD(true, pgram), \
        MAD_rel(true, pgram), RMS(true, pgram)


def MAD(true, y):
    return np.median(np.abs(y - true))

def MAD_rel(true, y):
    return np.median(np.abs(y - true)/true) * 100

def RMS(true, y):
    return (np.mean(true - y)**2)**.5


if __name__ == "__main__":

    truths = pd.read_csv("truths_extended_02_03.csv")
    truths = pd.read_csv("truths_extended_02_17.csv")

    MAD_ln_acf, MAD_acf, MAD_rel_acf, acfRMS = acf_plot(truths)
    MAD_ln_pgram, MAD_pgram, MAD_rel_pgram, RMS = pgram_plot(truths)
    print("ln(acf) MAD = ", MAD_ln_acf)
    print("acf MAD = ", MAD_acf)
    print("acf relative MAD = ", MAD_rel_acf)
    print("acf RMS = ", acfRMS)
    print("ln(pgram) MAD = ", MAD_ln_pgram)
    print("pgram MAD = ", MAD_pgram)
    print("pgram relative MAD = ", MAD_rel_pgram)
    print("pgram RMS = ", RMS)
