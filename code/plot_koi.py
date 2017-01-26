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


def load_and_plot(data, DATA_DIR, RESULTS_DIR):

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
    l = (recovered > 0) * (data.koi_id.values != 197) * \
        (data.koi_id.values != 279)


    plt.clf()
    xs = np.log(np.linspace(0, 100, 100))
    plt.plot(xs, xs, "-", color=".7", zorder=0)
    plt.plot(xs, xs + 2./3, "--", color=".7", zorder=0)
    plt.plot(xs, xs - 2./3, "--", color=".7", zorder=0)

    plt.errorbar(np.log(x[l]), np.log(recovered[l]), yerr=[lnerrp[l],
                 lnerrm[l]], xerr=lnxerr[l], fmt="k.", zorder=1, capsize=0,
                 ecolor=".7", alpha=.5, ms=.1)
    plt.scatter(np.log(x[l]), np.log(recovered[l]), c=np.log(amp[l]),
                edgecolor=".5", cmap="GnBu_r", vmin=min(np.log(amp[l])),
                vmax=max(np.log(amp[l])), s=20, zorder=2, lw=.2)

    import kplr
    import kepler_data as kd
    client = kplr.API()
    m = 20 < (np.abs(x[l] - recovered[l])/x[l] * 100)
    for i, koi in enumerate(data.koi_id.values):
        print(data.koi_id.values[l][m][i], x[l][m][i], recovered[l][m][i])
        star = client.koi("{}.01".format(str(data.koi_id.values[l][m][i])))
        # star.get_light_curves(fetch=True, shortcadence=False)
        print(star.kepid)
        kid = str(int(star.kepid)).zfill(9)
        fname = "/Users/ruthangus/.kplr/data/lightcurves/{}".format(kid)
        x, y, yerr = kd.load_kepler_data(fname)
        plt.clf()
        plt.plot(x, y, "k.")
        plt.xlim(150, 300)
        plt.savefig("{}".format(data.koi_id.values[l][m][i]))

    plt.ylim(0, 4)
    plt.xlim(0, 4)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln(\mathrm{R}_{\mathrm{var}})$")
    plt.xlabel("$\ln(\mathrm{McQuillan~{\it et~al.,}~2013~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    plt.savefig("comparison_koi")
    plt.savefig("../documents/figures/comparison_koi.pdf")

    print((np.median((x[l] - recovered[l])**2))**.5)


if __name__ == "__main__":
    RESULTS_DIR = "koi_results_01_23"
    DATA_DIR = "data"
    data = pd.read_csv(os.path.join(DATA_DIR, "Table_1.txt"), skiprows=19)
    load_and_plot(data, DATA_DIR, RESULTS_DIR)
