# Make plots of real Kepler light curve with model fit and corner plot

import os
import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd

import kplr
import kepler_data as kd
import george
from george.kernels import ExpSquaredKernel, ExpSine2Kernel, WhiteKernel

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def download_lc(koi):
    """
    Download Kepler data.
    """
    client = kplr.API()
    star = client.koi("{}.01".format(koi))
    print(star.kepid)
    star.get_light_curves(fetch=True, short_cadence=False)


def MAP(samples):
    hist, bins = np.histogram(samples, 100)
    return bins[hist == max(hist)][0]


def plot_lc(koi):
    """
    Make demo plot of a light curve.
    """

    # Load the data
    print(LC_DIR)
    x, y, yerr = kd.load_kepler_data(LC_DIR)
    x -= x[0]
    m = x < 500
    x, y, yerr = x[m], y[m], yerr[m]

    # Load the posterior samples.
    df = pd.read_hdf(os.path.join(DATA_DIR, "KOI-{}.h5".format(int(koi))),
                     key="samples")
    a = np.exp(MAP(df.ln_A.values))
    l = np.exp(MAP(df.ln_l.values))
    g = np.exp(MAP(df.ln_G.values))
    s = np.exp(MAP(df.ln_sigma.values))
    p = np.exp(MAP(df.ln_period.values))
    print("ln(a) = ", np.log(a), "ln(l) = ", np.log(l), "ln(G) = ", np.log(g),
          "ln(s) = ", np.log(s), "ln(p) = ", np.log(p), "p = ", p)

    xs = np.linspace(min(x), max(x), 500)
    k = a * ExpSquaredKernel(l) \
        * ExpSine2Kernel(g, p) + WhiteKernel(s)
    gp = george.GP(k)
    gp.compute(x, yerr)
    mu, cov = gp.predict(y, xs)

    plt.clf()
    plt.plot(x, y, "k.")
    plt.plot(xs, mu, color="CornFlowerBlue")
    plt.xlabel("$\mathrm{Time~(days)}$")
    plt.ylabel("$\mathrm{Normalised~flux}$")
    plt.subplots_adjust(left=.18)
    plt.savefig(os.path.join(FIG_DIR, "koi_lc_demo.pdf"))


if __name__ == "__main__":

    FIG_DIR = "/Users/ruthangus/projects/GProtation/documents/figures"
    DATA_DIR = "/Users/ruthangus/projects/GProtation/code/koi_results_01_23"

    koi = 1050
    KID = 5809890
    LC_DIR = "/Users/ruthangus/.kplr/data/lightcurves/" \
        "{}".format(str(int(KID)).zfill(9))

    # download_lc(koi)
    plot_lc(koi)
