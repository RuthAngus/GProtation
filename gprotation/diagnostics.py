from __future__ import print_function
import numpy as np
from GProt import calc_p_init, mcmc_fit
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def make_new_df(truths, R_DIR):
    """
    Load all the resulting period measurements and make a new pandas
    dataframe.
    """
    m = truths.DELTA_OMEGA.values == 0

    # get column names
    mfname2 = os.path.join(R_DIR, "0002_mcmc_results.csv")
    apfname2 = os.path.join(R_DIR, "0002_acf_pgram_results.csv")
    mdf2, adf2 = pd.read_csv(mfname2), pd.read_csv(apfname2)

    # assemble master data frame
    mcols, acols = mdf2.columns.values, adf2.columns.values
    mcmc = pd.DataFrame(data=np.zeros((0, len(mcols))), columns=mcols)
    acf_pgram = pd.DataFrame(data=np.zeros((0, len(acols))), columns=acols)
    Ns = []
    for i, id in enumerate(truths.N.values[m]):
        sid = str(int(id)).zfill(4)
        mfname = os.path.join(R_DIR, "{0}_mcmc_results.csv".format(sid))
        afname = os.path.join(R_DIR, "{0}_acf_pgram_results.csv".format(sid))
        if os.path.exists(mfname) and os.path.exists(afname):
            Ns.append(int(sid))
            mcmc = pd.concat([mcmc, pd.read_csv(mfname)], axis=0)
            acf_pgram = pd.concat([acf_pgram, pd.read_csv(afname)], axis=0)

    mcmc["N"], acf_pgram["N"] = np.array(Ns), np.array(Ns)
    truths1 = mcmc.merge(acf_pgram, on="N")
    truths_s = truths.merge(truths1, on="N")
    truths_s.to_csv("truths_extended.csv")
    return truths_s


def plots(truths, DIR):
    """
    Plot the GP and acf results.
    """

    truths_e = make_new_df(truths, DIR)
    m = (truths_e.DELTA_OMEGA.values == 0) \
            * (truths_e.acf_period.values > 0)

    N = truths_e.N.values[m]
    true = truths_e.P_MIN.values[m]
    med = np.exp(truths_e.sigma.values[m])  # period and sigma names swapped
    med_errp = np.exp(truths_e.sigma_errp.values[m])
    med_errm = np.exp(truths_e.sigma_errm.values[m])
    maxlike = np.exp(truths_e.sigma_max.values[m])
    amp = truths_e.AMP.values[m]
    acfs = truths_e.acf_period.values[m]
    acf_errs = truths_e.acf_period_err.values[m]

    mgp = (np.abs(true - maxlike) / true) < .1
    mgpf = (np.abs(true - maxlike) / true) > .2
    ma = (np.abs(true - acfs) / true) < .1
    maf = (np.abs(true - acfs) / true) > .2

    # plot mcmc results for acf successes
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.title("$\mathrm{MCMC~results~for~ACF~successes}$")
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.scatter(true[ma], maxlike[ma], c=np.log(amp[ma]), edgecolor="",
                cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "mcmc_acf"))

    # plot mcmc results for acf failures
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.title("$\mathrm{MCMC~results~for~ACF~failures}$")
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.scatter(true[maf], maxlike[maf], c=np.log(amp[maf]), edgecolor="",
                cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "mcmc_acf_fail"))

    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.title("$\mathrm{ACF~successes}$")
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.scatter(true[ma], acfs[ma], c=np.log(amp[ma]), edgecolor="",
                cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "acf"))

    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.title("$\mathrm{ACF~failures}$")
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.scatter(true[maf], acfs[maf], c=np.log(amp[maf]), edgecolor="",
                cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "acf_fail"))

    # plot acf results for mcmc successes
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.title("$\mathrm{ACF~results~for~MCMC~successes}$")
    plt.scatter(true[mgp], acfs[mgp], c=np.log(amp[mgp]), edgecolor="",
                cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "acf_mcmc"))

    # plot acf results for mcmc failures
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.title("$\mathrm{ACF~results~for~MCMC~failures}$")
    plt.scatter(true[mgpf], acfs[mgpf], c=np.log(amp[mgpf]), edgecolor="",
                cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "acf_mcmc_fail"))

    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.title("$\mathrm{MCMC~successes}$")
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.scatter(true[mgp], maxlike[mgp], c=np.log(amp[mgp]), edgecolor="",
                cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "mcmc"))

    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.title("$\mathrm{MCMC~failures}$")
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.scatter(true[mgpf], maxlike[mgpf], c=np.log(amp[mgpf]), edgecolor="",
                cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "mcmc_fail"))


if __name__ == "__main__":

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")

    print("mcmc Gprior rms = ", plots(truths, "results_initialisation"))
