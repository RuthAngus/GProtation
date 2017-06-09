
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from make_acf_comparison_plot import get_bad_inds


plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def plot(prior, nbins):
    """
    Make the two GP comparison plots for the paper --- one with an ACF prior
    (prior = True) and one without (prior = False).
    params:
    ------
    prior: (bool)
        If True the ACF prior results are plotted, if False the no prior
        results are plotted.
    nbins: (int)
        The number of bins to use to find the MAP.
    """

    if prior:
        RESULTS_DIR = "results_acfprior_03_10"
    else:
        RESULTS_DIR = "results_noprior_02_13"
    truths = pd.read_csv("final_table.txt", delimiter=" ")

    # remove differential rotators and take just the first 100
    m = truths.DELTA_OMEGA.values == 0
    truths = truths.iloc[m]
    N = truths.N.values

    recovered = np.zeros(len(truths.N.values))
    errp, errm = [np.zeros(len(truths.N.values)) for i in range(2)]
    lnerrp, lnerrm = [np.zeros(len(truths.N.values)) for i in range(2)]
    for i, id in enumerate(truths.N.values):
        fn = os.path.join(RESULTS_DIR, "{}.h5".format(id))
        if os.path.exists(fn):
            # store = pd.HDFStore(fn)
            # print(store)
            df = pd.read_hdf(fn, key="samples")
            phist, bins = np.histogram(df.ln_period.values, nbins)
            ln_p = bins[phist == max(phist)][0]
            # ln_p = np.median(df.ln_period.values)
            recovered[i] = np.exp(ln_p)
            lnerrp[i] = np.percentile(df.ln_period.values, 84) - ln_p
            lnerrm[i] = ln_p - np.percentile(df.ln_period.values, 16)
            errp[i] = np.exp(lnerrp[i]/ln_p)
            errm[i] = np.exp(lnerrm[i]/ln_p)

    x = .5 * (truths.P_MIN.values + truths.P_MAX.values)
    amp = truths.AMP.values
    l = recovered > 0

    plt.clf()
    xs = np.log(np.linspace(0, 55, 100))
    plt.plot(xs, xs, "-", color=".7", lw=.8, zorder=0)
    plt.plot(xs, xs + 2./3, "--", color=".7", lw=.8, zorder=0)
    plt.plot(xs, xs - 2./3, "--", color=".7", lw=.8, zorder=0)

    plt.errorbar(np.log(x[l]), np.log(recovered[l]),
                 yerr=[lnerrp[l], lnerrm[l]], fmt="k.", zorder=1, capsize=0,
                 ecolor=".2", alpha=.4, ms=.1, elinewidth=.8)
    plt.scatter(np.log(x[l]), np.log(recovered[l]), c=np.log(amp[l]),
                edgecolor=".5", cmap="GnBu_r", vmin=min(np.log(amp[l])),
                vmax=max(np.log(amp[l])), s=10, zorder=2, lw=.2)

    # badinds, varinds = get_bad_inds(N[l])
    # plt.plot(np.log(x[l])[badinds], np.log(recovered[l])[badinds], "ro")
    # plt.plot(np.log(x[l])[varinds], np.log(recovered[l])[varinds], "bo")

    plt.xlim(0, 4)
    plt.ylim(-2, 6)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    plt.subplots_adjust(bottom=.15)
    if prior:
        plt.savefig("comparison_acfprior_02_03")
        plt.savefig(os.path.join(FIG_DIR, "comparison_acfprior_03_10.pdf"))
    else:
        plt.savefig("comparison_noprior")
        plt.savefig(os.path.join(FIG_DIR, "comparison_noprior_02_13.pdf"))

    plt.clf()
    resids = np.log(x[l]) - np.log(recovered[l])
    linresids = x[l] - recovered[l]
    linerrs = .5*(lnerrp[l]*recovered[l] + lnerrm[l]*recovered[l])
    plt.hist(resids, 80, histtype="stepfilled", color="w")
    plt.xlabel("$\ln(\mathrm{True~Period}) - \ln(\mathrm{GP~Period})$")
    plt.axvline(np.percentile(resids, 16), color=".5", ls="--")
    plt.axvline(np.percentile(resids, 84), color=".5", ls="--")

    median_err = .5*(np.median(lnerrp[l]) + np.median(lnerrm[l]))
    median_err = np.median(.5*(lnerrp[l] + lnerrm[l]))
    plt.errorbar(-1, 100, xerr=median_err, fmt="k.", ms=.1)
    if prior:
        plt.savefig("gp_hist.pdf")
    else:
        plt.savefig("gp_hist_noprior.pdf")

    print(len(x[l]), "stars")
    print("MAD = ", np.median(np.abs(x[l] - recovered[l])))
    print("MAD (log) = ", np.median(np.abs(np.log(x[l]) -
                                           np.log(recovered[l]))))
    print("MAD relative % = ", np.median((np.abs(x[l] -
                                                 recovered[l]))/x[l])*100)
    print("RMS = ", (np.mean((x[l] - recovered[l])**2))**.5)

    errs = .5*(lnerrp[l] + lnerrm[l])
    plt.clf()
    plt.hist(errs, 100)
    if prior:
        plt.savefig("err_hist")
    else:
        plt.savefig("err_hist_noprior")

    plt.clf()
    nsigma_diff = np.abs(resids - errs)/errs
    plt.hist(nsigma_diff, 100)
    # plt.axvline(np.percentile(nsigma_diff, 66), color="r", ls="--")
    print(np.percentile(nsigma_diff, 50), "= median")
    "50% are 1.88 sigma off. But you'd expect "
    print(np.percentile(nsigma_diff, 66))
    print(max(nsigma_diff))
    if prior:
        plt.savefig("err_resid_ratio_hist")
    else:
        plt.savefig("err_resid_ratio_hist_noprior")

    if prior:
        plt.clf()
        print(np.std(linresids))
        plt.hist(linresids, 100)
        plt.savefig("resid_hist")
        print(np.mean(linerrs/recovered[l]),
              np.median(linerrs/recovered[l]))


    """
    3/4 of uncertainties are under-estimated.
    1/2 are within 2 sigma.
    66% are within 3 sigma.
    Largest outlier is 114 sigma off.
    """

    dff = np.log(recovered[l]) - np.log(x[l])
    plt.clf()
    m = dff > 1
    m = dff < -.5
    plt.plot(np.log(x[l]), dff, "k.")
    print(N[m])
    plt.savefig("diff_gp")

    print((np.median(np.abs(recovered[l] - x[l]))))


if __name__ == "__main__":
    FIG_DIR = "/Users/ruthangus/projects/GProtation/documents/figures"
    print("ACF prior")
    nbins = 95
    plot(True, nbins)  # with ACF prior
    print("\n", "No prior")
    plot(False, nbins)  # without ACF prior
