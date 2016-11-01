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

    # get column names
    mfname2 = os.path.join(R_DIR, "0002_mcmc_results.csv")
    apfname2 = os.path.join(R_DIR, "0002_acf_pgram_results.csv")
    mdf2, adf2 = pd.read_csv(mfname2), pd.read_csv(apfname2)

    # assemble master data frame
    mcols, acols = mdf2.columns.values, adf2.columns.values
    mcmc = pd.DataFrame(data=np.zeros((0, len(mcols))), columns=mcols)
    acf_pgram = pd.DataFrame(data=np.zeros((0, len(acols))), columns=acols)
    Ns = []
    try:
        N = truths.N.values
    except AttributeError:
        N = np.arange(0, np.shape(truths)[0])
    for i, id in enumerate(N):
        sid = str(int(id)).zfill(4)
        mfname = os.path.join(R_DIR, "{0}_mcmc_results.csv".format(sid))
        afname = os.path.join(R_DIR, "{0}_acf_pgram_results.csv".format(sid))
        if os.path.exists(mfname) and os.path.exists(afname):
            Ns.append(int(sid))
            mcmc = pd.concat([mcmc, pd.read_csv(mfname)], axis=0)
            acf_pgram = pd.concat([acf_pgram, pd.read_csv(afname)], axis=0)

    mcmc["N"], acf_pgram["N"] = np.array(Ns), np.array(Ns)
    truths1 = mcmc.merge(acf_pgram, on="N")
    try:
        truths_s = truths.merge(truths1, on="N")
    except KeyError:
        truths["N"] = np.arange(np.shape(truths)[0])
        truths_s = truths.merge(truths1, on="N")
    truths_s.to_csv("truths_extended{0}.csv".format(R_DIR))
    return truths_s


def mcmc_plots(truths_e, N, true, med, med_errp, med_errm, maxlike, amp, DIR):
    """
    Plot the GP results.
    """

    plt.clf()
    plt.plot(truths_e.NSPOT, np.exp(truths_e.gamma_max), "k.")
    plt.xlabel("nspot")
    plt.ylabel("gamma")
    plt.ylim(0, 5)
    plt.savefig("tau_gamma")

    plt.clf()
    xs = np.arange(0, 100, 1)
#     plt.plot(xs, xs, "k--", alpha=.5)
#     plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.ylim(0, 100)
#     plt.ylim(0, 1)
    plt.xlim(0, 55)
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
#     plt.errorbar(true, maxlike, yerr=[med_errp, med_errm], fmt="k.", zorder=0,
#                  capsize=0)
    plt.scatter(true, maxlike, c=np.log(amp), edgecolor="", cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "compare_mcmc"))

    # mcmc plot

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
    return (np.median((maxlike - true)**2))**.5


def acf_plot(truths, N, true, acfs, acf_errs, DIR):
    """
    Plot the acf results.
    """
    # acf plot
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.errorbar(true, acfs, yerr=acf_errs, fmt="k.", capsize=0, ecolor=".7",
                 zorder=0)
    plt.scatter(true, acfs, c=np.log(amp), edgecolor="", cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.ylim(0, 100)
    plt.xlim(0, 55)
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.ylabel("$\mathrm{Recovered~Period~(Days)}$")
    plt.savefig(os.path.join(DIR, "compare_acf"))
    return (np.median((acfs - true)**2))**.5

def pgram_plot(truths_e, N, true, pgram, DIR):
    """
    Plot the pgram results.
    """

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

#     gp_plots("periodic_gp_simulations")
#     assert 0

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    R_DIR = "results_Gprior"
    truths_e = make_new_df(truths, R_DIR)
    m = (truths_e.DELTA_OMEGA.values == 0) \
            * (truths_e.acf_period.values > 0)

    N = truths_e.N.values
    true = truths_e.P_MIN.values
    med = np.exp(truths_e.sigma.values)  # period and sigma names swapped
    med_errp = np.exp(truths_e.sigma_errp.values)
    med_errm = np.exp(truths_e.sigma_errm.values)
    maxlike = np.exp(truths_e.sigma_max)
    pgram = truths_e.pgram_period.values
    amp = truths_e.AMP.values
    acfs = truths_e.acf_period.values
    acf_errs = truths_e.acf_period_err.values

#     print("mcmc Gprior rms = ", mcmc_plots(truths_e, N, true, med, med_errp,
#           med_errm, maxlike, amp, R_DIR))

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    R_DIR = "results"
    m = truths.DELTA_OMEGA.values == 0
    truths_e = make_new_df(truths[m], R_DIR)
    m = (truths_e.DELTA_OMEGA.values == 0) \
            * (truths_e.acf_period.values > 0)

    N = truths_e.N.values
    true = truths_e.P_MIN.values
    med = np.exp(truths_e.sigma.values)  # period and sigma names swapped
    med_errp = np.exp(truths_e.sigma_errp.values)
    med_errm = np.exp(truths_e.sigma_errm.values)
    maxlike = np.exp(truths_e.sigma_max)
    pgram = truths_e.pgram_period.values
    amp = truths_e.AMP.values
    acfs = truths_e.acf_period.values
    acf_errs = truths_e.acf_period_err.values

#     print("mcmc rms = ", mcmc_plots(truths_e, N, true, med, med_errp,
#            med_errm, maxlike, amp, "results"))
#
#     print("acf rms = ", acf_plot(truths_e, N, true, acfs, acf_errs, "results"))
#     print("pgram rms = ", pgram_plot(truths_e, N, true, pgram, "results"))

#    print("mcmc prior rms = ", mcmc_plots(truths_e, N, true, med, med_errp,
#           med_errm, maxlike, amp, "results_prior"))

    gp_truths = pd.read_csv("gp_truths.csv")

    truths_e_gp = make_new_df(gp_truths, "results_periodic_gp")
    N = truths_e_gp.N.values
    true = np.exp(truths_e_gp.lnperiod.values)
    med = np.exp(truths_e_gp.sigma.values)
    med_errp = np.exp(truths_e_gp.sigma_errp.values)
    med_errm = np.exp(truths_e_gp.sigma_errm.values)
    maxlike = np.exp(truths_e_gp.sigma_max.values)
    amp = np.ones_like(maxlike)
    print(maxlike)
    truths_e_gp["NSPOT"] = np.ones_like(maxlike)
    print("periodic rms = ", mcmc_plots(truths_e_gp, N, true, med, med_errp,
          med_errm, maxlike, amp, "results_periodic_gp"))
