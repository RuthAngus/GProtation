from __future__ import print_function
import numpy as np
from GProt import calc_p_init, mcmc_fit
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py

from calc_mad import MAD, MAD_rel, RMS

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)

DATA_DIR = "Users/ruthangus/projects/GProtation/" \
    "code/kepler_diffrot_full/final"

def calc_p_init(x, y, yerr, id, RESULTS_DIR):
    fname = os.path.join(RESULTS_DIR, "{0}_acf_pgram_results.txt".format(id))
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


def load_samples(id, RESULTS_DIR):
    try:
        fname = os.path.join(RESULTS_DIR,
                             "{0}.h5".format(str(int(id)).zfill(4)))
    except IOError:
        fname = os.path.join(RESULTS_DIR,
                             "{0}.h5".format(str(int(id)).zfill(9)))

    if not os.path.exists(fname):
        return None
    with h5py.File(fname, "r") as f:
        samples = f["samples"][...]
    if len(np.shape(samples)) < 3:  # if emcee3
        return samples[:, 4]
    nwalkers, nsteps, ndims = np.shape(samples)
    return np.reshape(samples[:, :, 4], nwalkers * nsteps)


def make_new_df(truths, R_DIR):
    """
    Load all the resulting period measurements and make a new pandas
    dataframe.
    """
    m = truths.DELTA_OMEGA.values == 0

    # get column names
    mfname2 = os.path.join(R_DIR, "0002_mcmc_results.txt")
    if os.path.exists(mfname2):
        apfname2 = os.path.join(R_DIR, "2_acf_pgram_results.txt")
        mdf2, adf2 = pd.read_csv(mfname2), pd.read_csv(apfname2)

    # assemble master data frame
    mcols, acols = mdf2.columns.values, adf2.columns.values
    mcmc = pd.DataFrame(data=np.zeros((0, len(mcols))), columns=mcols)
    acf_pgram = pd.DataFrame(data=np.zeros((0, len(acols))), columns=acols)
    Ns = []
    for i, id in enumerate(truths.N.values[m]):
        sid4 = str(int(id)).zfill(4)
        sid9 = str(int(id))
        mfname = os.path.join(R_DIR, "{0}_mcmc_results.txt".format(sid4))
        afname = os.path.join(R_DIR, "{0}_acf_pgram_results.txt".format(sid9))
        if os.path.exists(mfname) and os.path.exists(afname):
            Ns.append(int(sid9))
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
    acorr_A = truths_e.acorr_A.values[m]
    acorr_l = truths_e.acorr_l.values[m]
    acorr_g = truths_e.acorr_gamma.values[m]
    acorr_s = truths_e.acorr_sigma.values[m]
    acorr_p = truths_e.acorr_period.values[m]
    acorr = np.vstack((acorr_A, acorr_l, acorr_g, acorr_s, acorr_p))
    mean_acorr = np.mean(acorr, axis=0)

    # mcmc plot
    plt.clf()
    xs = np.log(np.arange(0, 100, 1))
    plt.plot(xs, xs, "k-", alpha=.2, zorder=0)
    plt.plot(xs, xs + 2./3, "k--", alpha=.2, zorder=0)
    plt.plot(xs, xs - 2./3, "k--", alpha=.2, zorder=0)
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period,~GP~Method})$")

    print(len(true))
    plt.errorbar(np.log(true), np.log(maxlike), yerr=[lnerrp, lnerrm],
                 fmt="k.", zorder=1, capsize=0, ecolor=".8", alpha=.5, ms=.1)
    plt.scatter(np.log(true), np.log(maxlike), c=np.log(amp), edgecolor=".5",
                cmap="GnBu_r", vmin=min(np.log(amp)), vmax=max(np.log(amp)),
                s=20, lw=.2, zorder=2)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    # plt.savefig(os.path.join(DIR, "compare_mcmc.pdf"))

    # make convergence plot.
    plt.clf()
    plt.plot(xs, xs, "k-", alpha=.2, zorder=0)
    plt.plot(xs, xs + 2./3, "k--", alpha=.2, zorder=0)
    plt.plot(xs, xs - 2./3, "k--", alpha=.2, zorder=0)
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    plt.errorbar(np.log(true), np.log(maxlike), yerr=[lnerrp, lnerrm],
                 fmt="k.", zorder=1, capsize=0, ecolor=".8", alpha=.5, ms=.1)
    plt.scatter(np.log(true), np.log(maxlike), c=mean_acorr, edgecolor="k",
                cmap="GnBu_r", vmin=min(mean_acorr), vmax=max(mean_acorr),
                s=20, lw=.2, zorder=2)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$<\mathrm{(Autocorrelation~Time)}>$")
    # plt.savefig(os.path.join(DIR, "compare_mcmc_convergence"))


    # mcmc plot with samples
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(np.log(xs), np.log(xs), "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) + 2./3, "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) - 2./3, "k--", alpha=.5)
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    for i, n in enumerate(N):
        samples = load_samples(n, DIR)
        if samples != None:
            plt.plot(np.log(np.ones(100) * true[i]), np.random.choice(samples,
                     100), "k.", ms=1)
    # plt.savefig(os.path.join(DIR, "compare_mcmc_samples.pdf"))
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

    print(len(true))
    # acf plot
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(np.log(xs), np.log(xs), "k-", alpha=.3, zorder=0)
    plt.plot(np.log(xs), np.log(xs) - 2./3, "k--", alpha=.3, zorder=0)
    plt.plot(np.log(xs), np.log(xs) + 2./3, "k--", alpha=.3, zorder=0)

    # plt.errorbar(np.log(true), np.log(acfs), yerr=(acf_errs/acfs), fmt="k.",
                 # capsize=0, ecolor=".7", alpha=.4, ms=1, zorder=1)
    plt.scatter(np.log(true), np.log(acfs), c=np.log(amp), edgecolor=".5",
                cmap="GnBu_r", vmin=min(np.log(amp)), vmax=max(np.log(amp)),
                s=20, lw=.2, zorder=2)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period~ACF~Method})$")
    # plt.savefig(os.path.join(DIR, "compare_acf.pdf"))

    return np.median(np.abs(acfs - true)), np.log(true) - np.log(acfs)

def pgram_plot(truths, DIR):
    """
    Plot the pgram results.
    """

    truths_e = make_new_df(truths, DIR)
    m = (truths_e.DELTA_OMEGA.values == 0) #\
            # * (truths_e.acf_period.values > 0)

    bad = np.array([121, 164, 180, 188, 189, 227, 239, 246, 253, 269, 273,
                    328, 336, 338, 361, 377, 397, 470, 479, 556, 573, 608,
                    615, 671])
    variable = np.array([345, 350, 356, 370, 406, 477, 450, 663, 682, 692])
    badinds, varinds = [], []

    N = truths_e.N.values[m]
    for i, n in enumerate(N):
        mb = n == bad
        mv = n == variable
        if len(bad[mb]):
            badinds.append(i)
        if len(variable[mv]):
            varinds.append(i)

    true = truths_e.P_MIN.values[m]
    pgram = truths_e.pgram_period.values[m]
    pgram_err = truths_e.pgram_period_err.values[m]
    amp = truths_e.AMP.values[m]

    print(len(true))
    # pgram plot
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(np.log(xs), np.log(xs), "k-", alpha=.3, zorder=0)
    plt.plot(np.log(xs), np.log(xs) - 2./3, "k--", alpha=.3, zorder=0)
    plt.plot(np.log(xs), np.log(xs) + 2./3, "k--", alpha=.3, zorder=0)
    plt.errorbar(np.log(true), np.log(pgram), yerr=pgram_err, fmt="k.",
                 zorder=0, ecolor=".2", alpha=.4, ms=1, elinewidth=.8)
    plt.scatter(np.log(true), np.log(pgram), c=np.log(amp), edgecolor=".5",
                cmap="GnBu_r", vmin=min(np.log(amp)), vmax=max(np.log(amp)),
                s=10, zorder=1, lw=.2)
    # plt.plot(np.log(true)[badinds], np.log(pgram)[badinds], "ro")
    # plt.plot(np.log(true)[varinds], np.log(pgram)[varinds], "bo")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.xlim(0, 4)
    plt.ylim(-2, 6)
    plt.xlabel("$\mathrm{Injected~Period~(Days)}$")
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period~LS~Periodogram~method})$")
    plt.subplots_adjust(bottom=.15)
    print(os.path.join(DIR, "compare_pgram_filtered_35.pdf"))
    plt.savefig(os.path.join(DIR, "compare_pgram_filtered_35.pdf"))
    plt.savefig("../documents/figures/compare_pgram_filtered_35.pdf")

    dff = np.log(pgram) - np.log(true)
    plt.clf()
    m = dff > 1
    plt.plot(np.log(true), dff, "k.")
    print(N[m])
    plt.savefig("diff")

    print("ln MAD = ", MAD(np.log(true), np.log(pgram)), "MAD = ",
          MAD(true, pgram), "MAD_rel = ", MAD_rel(true, pgram),
          "RMS = ", RMS(true, pgram))

    return np.median(np.abs(pgram - true)), np.log(true) - np.log(pgram)


if __name__ == "__main__":

    DIR = "/Users/ruthangus/projects/GProtation/code/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    pgram_MAD, pgram_resids = \
        pgram_plot(truths, "../gprotation/pgram_filtered_results_35")
    # pgram_MAD, pgram_resids = \
    #     pgram_plot(truths, "../gprotation/pgram_results")

    # remove 17 for now
#     m = truths.N.values != 17
#     truths = truths.iloc[m]

    # acf_MAD, acf_resids = acf_plot(truths, "acf_results")
    # pgram_MAD, pgram_resids = pgram_plot(truths, "acf_results")
    # print("acf MAD = ", acf_MAD)
    # print("pgram MAD = ", pgram_MAD)
    # m1 = (-.5 < acf_resids) * (acf_resids < .5)
    # m2 = (-.5 < pgram_resids) * (pgram_resids < .5)
    # plt.clf()
    # plt.hist(pgram_resids[m2], 100, normed=True, alpha=.5, color="r",
    #          label="pgram")
    # plt.hist(acf_resids[m1], 100, normed=True, alpha=.5, color="b",
    #          label="ACF")
    # print(np.median(np.abs(acf_resids)))
    # print(np.median(np.abs(pgram_resids)))
    # plt.legend()
    # plt.xlim(-1, 1)
    # plt.savefig("acf_pgram_resids")
