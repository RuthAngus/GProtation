import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import simple_acf as sa
from acf import Kepler_ACF

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def lc_demo(x, y):
    plt.clf()
    plt.plot(x, y, "ko", ms=1)
    plt.xlabel("$\mathrm{Time~(Days)}$")
    plt.ylabel("$\mathrm{Normalised~Flux}$")
    plt.savefig(os.path.join(FIG_DIR, "demo_lc.pdf"))


def acf_demo(x, y):
    yerr = np.ones_like(y) * 1e-5
    period, err, lags, acf = Kepler_ACF.corr_run(x, y, yerr, 2)
    # period, acf, lags = sa.simple_acf(x, y)

    plt.clf()
    plt.plot(lags, acf, "k")
    plt.axvline(period, color="CornFlowerBlue")
    plt.xlabel("$\mathrm{Lags~(Days)}$")
    plt.ylabel("$\mathrm{Autocorrelation}$")
    plt.xlim(0, 470)
    plt.savefig(os.path.join(FIG_DIR, "demo_ACF.pdf"))


def period_hist():
    truths = pd.read_csv(os.path.join(DATA_DIR, "par/final_table.txt"),
                         delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0
    t = truths.iloc[m]

    plt.clf()
    plt.hist(t.P_MIN, histtype="stepfilled", color="w")
    plt.xlabel("$P_{\mathrm{rot}}~\mathrm{(Days)}$")
    plt.savefig(os.path.join(FIG_DIR, "period_hist.pdf"))

if __name__ == "__main__":
    DATA_DIR = "simulations/kepler_diffrot_full/"
    FIG_DIR = "/Users/ruthangus/projects/GProtation/documents/figures"
    x, y = np.genfromtxt(os.path.join(DATA_DIR,
                                      "final/lightcurve_0025.txt")).T
    lc_demo(x, y)
    acf_demo(x, y)
