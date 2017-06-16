# plot Tim's results files

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def load_and_plot(data):

    recovered, errp, errm, lnerrp, lnerrm = [np.zeros(len(data.koi_id.values))
                                             for i in range(5)]
    for i, kid in enumerate(data.koi_id.values):
        fn = os.path.join(RESULTS_DIR, "KOI-{}.h5".format(int(kid)))
        if os.path.exists(fn):
            df = pd.read_hdf(fn, key="samples")
            ln_p = np.median(df.ln_period.values)
            recovered[i] = np.exp(ln_p)
            lnerrp[i] = np.percentile(df.ln_period.values, 84) - ln_p
            lnerrm[i] = ln_p - np.percentile(df.ln_period.values, 16)
            errp[i] = np.exp(lnerrp[i]/ln_p)
            errm[i] = np.exp(lnerrm[i]/ln_p)

    x = data.P_rot.values
    xerr = data.P_rot_err.values
    lnxerr = xerr/x
    amp = data.R_var.values
    # remove binaries.
    l = np.isfinite(x) * (recovered > 0) * (data.koi_id.values != 197) * \
        (data.koi_id.values != 279)

    plt.clf()
    xs = np.log(np.linspace(0, 100, 100))
    plt.plot(xs, xs, "-", color=".7", lw=.8, zorder=0)
    plt.plot(xs, xs + 2./3, "--", color=".7", lw=.8, zorder=0)
    plt.plot(xs, xs - 2./3, "--", color=".7", lw=.8, zorder=0)

    print(lnerrp[l])
    assert 0
    plt.errorbar(np.log(x[l]), np.log(recovered[l]), yerr=[lnerrp[l],
                 lnerrm[l]], xerr=lnxerr[l], fmt="k.", zorder=1, capsize=0,
                 ecolor=".7", alpha=.5, ms=.1, elinewidth=.8)
    plt.scatter(np.log(x[l]), np.log(recovered[l]), c=np.log(amp[l]),
                edgecolor=".5", cmap="GnBu_r", vmin=min(np.log(amp[l])),
                vmax=max(np.log(amp[l])), s=10, zorder=2, lw=.2)

    plt.ylim(0, 4)
    plt.xlim(0, 4)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln(\mathrm{R}_{\mathrm{var}})$")
    plt.xlabel("$\ln(\mathrm{McQuillan~{\it et~al.,}~2013~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    plt.subplots_adjust(bottom=.15)
    plt.savefig("comparison_koi")
    plt.savefig(os.path.join(FIG_DIR, "comparison_koi_01_23.pdf"))

    print("MAD = ", np.median(np.abs(np.log(x[l]) - np.log(recovered[l]))))

    median_err = np.median(.5*(lnerrp[l] + lnerrm[l]))
    resids = np.log(x[l]) - np.log(recovered[l])
    print("percentiles:")
    print(np.percentile(resids, 16), np.percentile(resids, 84))
    print("diff = ",
        np.abs(np.percentile(resids, 16) - np.percentile(resids, 84)))
    print("median_err = ", median_err)
    print(np.median(lnxerr[np.isfinite(lnxerr)]), "mcquillan err")


    print("MAD = ", np.median(np.abs(x[l] - recovered[l])))
    print("MAD (log) = ", np.median(np.abs(np.log(x[l]) - np
                                           .log(recovered[l]))))
    print("MAD relative % = ", np.median((np.abs(x[l] -
                                                 recovered[l]))/x[l])*100)

#     errs = .5*(lnerrp[l] + lnerrm[l])
#     plt.clf()
#     plt.hist(errs, 100)
#     plt.savefig("err_hist_koi")

#     plt.clf()
#     nsigma_diff = np.abs(resids - errs)/errs
#     plt.hist(nsigma_diff, 100, histtype="stepfilled", color="w")
#     plt.axvline(np.percentile(nsigma_diff, 66), color="r", ls="--")
#     print(np.percentile(nsigma_diff, 66))
#     print(max(nsigma_diff))
#     plt.savefig("err_resid_ratio_hist_koi")

    """
    3/4 of uncertainties are under-estimated.
    1/2 are within 2 sigma.
    66% are within 3 sigma.
    Largest outlier is 114 sigma off.
    """

    print((np.median(np.abs(recovered[l] - x[l]))))


if __name__ == "__main__":
    path = "/Users/ruthangus/projects/GProtation"
    RESULTS_DIR = os.path.join(path, "code/koi_results_02_03")
    DATA_DIR = os.path.join(path, "code/data")
    FIG_DIR = os.path.join(path, "documents/figures")
    data = pd.read_csv(os.path.join(DATA_DIR, "Table_1.txt"), skiprows=19)
    load_and_plot(data)
