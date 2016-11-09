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

DATA_DIR = "../code/simulations/kepler_diffrot_full/final"
RESULTS_DIR = "results/"


def calc_p_init(x, y, yerr, id, RESULTS_DIR):
    fname = os.path.join(RESULTS_DIR, "{0}_acf_pgram_results.csv".format(id))
    if os.path.exists(fname):
        print("Previous ACF pgram result found")
        df = pd.read_csv(fname)
        m = df.N.values == int(id)
        acf_period = df.acf_period.values[m]
        err = df.acf_period_err.values[m]
        pgram_period = df.pgram_period.values[m]
        pgram_period_err = df.pgram_period_err.values[m]
    else:
        print("Calculating ACF")
        acf_period, err, lags, acf = corr_run(x, y, yerr, id, RESULTS_DIR)

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


def load_samples(id):
    fname = os.path.join(RESULTS_DIR, "{0}.h5".format(str(int(id)).zfill(4)))
    if os.path.exists(fname):
        with h5py.File(fname, "r") as f:
            samples = f["samples"][...]
    nwalkers, nsteps, ndims = np.shape(samples)
    return np.reshape(samples[:, :, 4], nwalkers * nsteps)


def make_new_df(truths, R_DIR):
    """
    Load all the resulting period measurements and make a new pandas
    dataframe.
    """
    m = truths.DELTA_OMEGA.values == 0

#     for i, id in enumerate(truths.N.values[m]):
#         sid = str(int(i)).zfill(4)
#         acf_period, a_err, pgram_period, p_err = calc_p_init(x, y, yerr, sid,
#                                                              RESULTS_DIR)

    # get column names
    mfname2 = os.path.join(R_DIR, "0002_mcmc_results.txt")
    apfname2 = os.path.join(R_DIR, "0002_acf_pgram_results.csv")
    mdf2, adf2 = pd.read_csv(mfname2), pd.read_csv(apfname2)

    # assemble master data frame
    mcols, acols = mdf2.columns.values, adf2.columns.values
    mcmc = pd.DataFrame(data=np.zeros((0, len(mcols))), columns=mcols)
    acf_pgram = pd.DataFrame(data=np.zeros((0, len(acols))), columns=acols)
    Ns = []
    n = 0
    for i, id in enumerate(truths.N.values[m]):
        sid = str(int(id)).zfill(4)
        mfname = os.path.join(R_DIR, "{0}_mcmc_results.txt".format(sid))
        afname = os.path.join(R_DIR, "{0}_acf_pgram_results.csv".format(sid))
        if os.path.exists(mfname) and os.path.exists(afname):
            n += 1
            Ns.append(int(sid))
            mcmc = pd.concat([mcmc, pd.read_csv(mfname)], axis=0)
            acf_pgram = pd.concat([acf_pgram, pd.read_csv(afname)], axis=0)

    mcmc["N"], acf_pgram["N"] = np.array(Ns), np.array(Ns)
    truths1 = mcmc.merge(acf_pgram, on="N")
    truths_s = truths.merge(truths1, on="N")
    truths_s.to_csv("truths_extended.csv")
    return truths_s


def mcmc_plots(truths, DIR):
    """
    Plot the GP results.
    """

    truths_e = make_new_df(truths, DIR)
    m = (truths_e.DELTA_OMEGA.values == 0) \
            * (truths_e.acf_period.values > 0)

    N = truths_e.N.values[m]
    true = truths_e.P_MIN.values[m]
    med = np.exp(truths_e.sigma.values[m])  # period and sigma names swapped
    med_errp = np.exp(truths_e.sigma_errp.values[m])
    lnp = truths_e.period[m]
    lnerrp = truths_e.period_errp.values[m]
    lnerrm = truths_e.period_errm.values[m]
    med_errm = np.exp(truths_e.sigma_errm.values[m])
    maxlike = np.exp(truths_e.sigma_max[m])
    amp = truths_e.AMP.values[m]

    plt.clf()
    plt.plot(truths_e.TAU / truths_e.P_MIN, np.exp(truths_e.gamma_max), "k.")
    plt.xlabel("tau / period")
    plt.ylabel("gamma")
    plt.ylim(0, 5)
    plt.savefig("tau_gamma")

    # mcmc plot
    plt.clf()
    xs = np.log(np.arange(0, 100, 1))
    plt.plot(xs, xs, "k-", alpha=.2, zorder=0)
    plt.plot(xs, xs + 2, "k--", alpha=.2, zorder=0)
    plt.plot(xs, xs + 2./3, "k--", alpha=.2, zorder=0)
    plt.plot(xs, xs - 2, "k--", alpha=.2, zorder=0)
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")

    plt.errorbar(np.log(true), np.log(maxlike), yerr=[lnerrp, lnerrm],
                 fmt="k.", zorder=1, capsize=0, ecolor=".8", alpha=.3, ms=.1)
    plt.scatter(np.log(true), np.log(maxlike), c=np.log(amp), edgecolor="",
                cmap="GnBu_r", vmin=min(np.log(amp)), vmax=max(np.log(amp)),
                s=15, zorder=2)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
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
    return (np.median((maxlike - true)**2))**.5, \
            (np.median((np.exp(lnp) - true)**2))**.5


def acf_plot(truths, DIR):
    """
    Plot the acf results.
    """
    truths_e = make_new_df(truths, DIR)
    m = (truths_e.DELTA_OMEGA.values == 0) \
            * (truths_e.acf_period.values > 0)

    N = truths_e.N.values[m]
    true = truths_e.P_MIN.values[m]
    acfs = truths_e.acf_period.values[m]
    acf_errs = truths_e.acf_period_err.values[m]
    amp = truths_e.AMP.values[m]

    # acf plot
    plt.clf()
#     xs = np.arange(0, 100, 1)
#     plt.plot(xs, xs, "k--", alpha=.5)
#     plt.plot(xs, 2*xs, "k--", alpha=.5)

#     plt.errorbar(true, acfs, yerr=acf_errs, fmt="k.", capsize=0, ecolor=".7",
#                  zorder=0)
#     plt.scatter(true, acfs, c=np.log(amp), edgecolor="", cmap="GnBu",
#                 vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    plt.errorbar(np.log(true), np.log(acfs), yerr=(acf_errs/acfs), fmt="k.",
                 capsize=0, ecolor=".7", zorder=0, alpha=.4)
#     plt.scatter(np.log(true), np.log(acfs), c=np.log(amp), edgecolor="",
#                 cmap="GnBu", vmin=min(np.log(amp)), vmax=max(np.log(amp)),
#                 s=50, zorder=1)
#     cbar = plt.colorbar()
#     cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
#     plt.ylim(0, 100)
#     plt.xlim(0, 55)
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.savefig(os.path.join(DIR, "compare_acf"))
    return (np.median((acfs - true)**2))**.5

def pgram_plot(truths, DIR):
    """
    Plot the pgram results.
    """

    truths_e = make_new_df(truths, DIR)
    m = (truths_e.DELTA_OMEGA.values == 0) \
            * (truths_e.acf_period.values > 0)

    N = truths_e.N.values[m]
    true = truths_e.P_MIN.values[m]
    pgram = truths_e.pgram_period.values[m]
    amp = truths_e.AMP.values[m]

    # pgram plot
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.scatter(true, pgram, c=np.log(amp), edgecolor="", cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.savefig(os.path.join(DIR, "compare_pgram"))
    return (np.median((pgram - true)**2))**.5

if __name__ == "__main__":

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")

#     print("mcmc Gprior rms = ", mcmc_plots(truths, "results_Gprior"))
    print("mcmc sigma rms = ", mcmc_plots(truths, "results_sigma"))
#     print("mcmc prior rms = ", mcmc_plots(truths, "results_prior"))
#     print("mcmc rms = ", mcmc_plots(truths, "results"))
#     print("acf rms = ", acf_plot(truths, "results_sigma"))
