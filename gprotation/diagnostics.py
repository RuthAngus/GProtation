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
    mfname2 = os.path.join(R_DIR, "0002_mcmc_results.txt")
    apfname2 = os.path.join(R_DIR, "0002_acf_pgram_results.csv")
    mdf2, adf2 = pd.read_csv(mfname2), pd.read_csv(apfname2)

    # assemble master data frame
    mcols, acols = mdf2.columns.values, adf2.columns.values
    mcmc = pd.DataFrame(data=np.zeros((0, len(mcols))), columns=mcols)
    acf_pgram = pd.DataFrame(data=np.zeros((0, len(acols))), columns=acols)
    Ns = []
    for i, id in enumerate(truths.N.values[m]):
        sid = str(int(id)).zfill(4)
        mfname = os.path.join(R_DIR, "{0}_mcmc_results.txt".format(sid))
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
    gammas = truths_e.gamma_max.values[m]
    ss = truths_e.sigma_max.values[m]
    As = truths_e.A_max.values[m]
    ls = truths_e.l_max.values[m]

    mgp = (np.abs(true - maxlike) / true) < .1
    mgpf = (np.abs(true - maxlike) / true) > .2
    ma = (np.abs(true - acfs) / true) < .1
    maf = (np.abs(true - acfs) / true) > .2

    # plot mcmc results for acf successes
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(xs, xs, "k--", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.plot(xs, .5*xs, "k--", alpha=.5)
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
    plt.plot(true[N == 44], acfs[N == 44], "r.")
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
    plt.plot(xs, xs, "k-", alpha=.5)
    plt.plot(xs, 2*xs, "k--", alpha=.5)
    plt.plot(xs, .5*xs, "k--", alpha=.5)
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
    m = ((maxlike[mgpf] / true[mgpf]) > 1.3)
#     * (maxlike[mgpf] > 40) \
#             * (maxlike[mgpf] < 60)
    print(N[mgpf][m])
    num = 3
    plt.plot(true[mgpf][m][num], maxlike[mgpf][m][num], "r.")
    print(N[mgpf][m][num])
    print(true[mgpf][m][num], maxlike[mgpf][m][num], acfs[mgpf][m][num])
    plt.savefig(os.path.join(DIR, "mcmc_fail"))

    plt.clf()
    plt.hist(gammas[mgpf], color="w", alpha=.5, histtype="stepfilled",
            ls="--",
            label="${0:.2} \pm {1:.2}$".format(np.median(gammas[mgpf]),
                                        np.std(gammas[mgpf])))
    plt.hist(gammas[mgp], color="w", alpha=.5, histtype="stepfilled",
            label="${0:.2} \pm {1:.2}$".format(np.median(gammas[mgp]),
                                        np.std(gammas[mgp])))
    plt.legend()
    plt.xlabel("$\ln(\Gamma)$")
    plt.savefig("gamma_comp")

    plt.clf()
    plt.hist(ls[mgpf], color="w", alpha=.5, histtype="stepfilled",
            ls="--",
            label="${0:.2} \pm {1:.2}$".format(np.median(ls[mgpf]),
                                        np.std(ls[mgpf])))
    plt.hist(ls[mgp], color="w", alpha=.5, histtype="stepfilled",
            label="${0:.2} \pm {1:.2}$".format(np.median(ls[mgp]),
                                        np.std(ls[mgp])))
    plt.legend()
    plt.xlabel("$\ln(l)$")
    plt.savefig("l_comp")

    plt.clf()
    plt.hist(As[mgpf], color="w", alpha=.5, histtype="stepfilled",
            ls="--",
            label="${0:.2} \pm {1:.2}$".format(np.median(As[mgpf]),
                                        np.std(As[mgpf])))
    plt.hist(As[mgp], color="w", alpha=.5, histtype="stepfilled",
            label="${0:.2} \pm {1:.2}$".format(np.median(As[mgp]),
                                        np.std(As[mgp])))
    plt.legend()
    plt.xlabel("$\ln(A)$")
    plt.savefig("A_comp")

    plt.clf()
    plt.hist(ss[mgpf], color="w", alpha=.5, histtype="stepfilled",
            ls="--",
            label="${0:.2} \pm {1:.2}$".format(np.median(ss[mgpf]),
                                        np.std(ss[mgpf])))
    plt.hist(ss[mgp], color="w", alpha=.5, histtype="stepfilled",
            label="${0:.2} \pm {1:.2}$".format(np.median(ss[mgp]),
                                        np.std(ss[mgp])))
    plt.legend()
    plt.xlabel("$\ln(\sigma)$")
    plt.savefig("s_comp")

    plt.clf()
    m = maxlike[mgpf] < 1000
    plt.hist(maxlike[mgpf][m], color="w", alpha=.5, histtype="stepfilled",
            ls="--",
            label="${0:.2} \pm {1:.2}$".format(np.median(maxlike[mgpf][m]),
                                        np.std(maxlike[mgpf][m])))
    plt.hist(maxlike[mgp], color="w", alpha=.5, histtype="stepfilled",
            label="${0:.2} \pm {1:.2}$".format(np.median(maxlike[mgp]),
                                        np.std(maxlike[mgp])))
    plt.legend()
    plt.xlabel("$\ln(\mathrm{Period})$")
    plt.savefig("period_comp")


if __name__ == "__main__":

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")

#     print("mcmc Gprior rms = ", plots(truths, "results_Gprior"))
    print("mcmc Gprior rms = ", plots(truths, "results_sigma"))
#     print("mcmc Gprior rms = ", plots(truths, "results_initialisation"))
