from __future__ import print_function
import numpy as np
from GProt import calc_p_init, mcmc_fit
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py

DATA_DIR = "../code/simulations/kepler_diffrot_full/final"
RESULTS_DIR = "results/"

def comparison_plot(truths, DIR):
    """
    Plot the acf, pgram and GP results.
    """

    plotpar = {'axes.labelsize': 18,
               'font.size': 10,
               'legend.fontsize': 15,
               'xtick.labelsize': 18,
               'ytick.labelsize': 18,
               'text.usetex': True}
    plt.rcParams.update(plotpar)

    truths_e = make_new_df(truths, DIR)
    m = (truths_e.DELTA_OMEGA.values == 0) \
            * (truths_e.acf_period.values > 0)

    N = truths_e.N.values[m]
    true = truths_e.P_MIN.values[m]
    acfs = truths_e.acf_period.values[m]
    acf_errs = truths_e.acf_period_err.values[m]
    pgram = truths_e.pgram_period.values[m]
    med = truths_e.med_mcmc_period.values[m]
    med_errp = truths_e.med_mcmc_period_errp.values[m]
    med_errm = truths_e.med_mcmc_period_errm.values[m]
    maxlike = truths_e.maxlike_mcmc_period[m]
    amp = truths_e.AMP.values[m]
    var = truths_e.variance.values[m]

    # mcmc plot
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.errorbar(true, maxlike, yerr=[med_errp, med_errm], fmt="k.", zorder=0,
                 capsize=0)
    plt.scatter(true, maxlike, c=np.log(var), edgecolor="", cmap="GnBu",
                vmin=min(np.log(var)), vmax=max(np.log(var)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Variance)}$")
    plt.savefig(os.path.join(DIR, "compare_mcmc"))

    # mcmc plot with samples
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    for i, n in enumerate(N):
        samples = load_samples(n)
        plt.plot(np.ones(100) * true[i], np.exp(np.random.choice(samples,
                                                                 100)),
                 "k.", ms=1)
    plt.savefig(os.path.join(DIR, "compare_mcmc_samples"))

    # acf plot
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.errorbar(true, acfs, yerr=acf_errs, fmt="k.", capsize=0, ecolor=".7",
                 alpha=.5)
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.savefig(os.path.join(DIR, "compare_acf"))

    # pgram plot
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.plot(true, pgram, "r.", alpha=.5)
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.savefig(os.path.join(DIR, "compare_pgram"))

if __name__ == "__main__":

    truths = pd.read_csv("truths_extended.csv")
    comparison_plot(truths, "results_prior")
    comparison_plot(truths, "results")
