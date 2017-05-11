"""
Plot the results of the GP fit.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSquaredKernel, ExpSine2Kernel, WhiteKernel
import corner
import pandas as pd


def load_suzanne_lcs(id, DATA_DIR="/Users/ruthangus/projects/GProtation/" \
                     "code/kepler_diffrot_full/final"):
    sid = str(int(id)).zfill(4)
    x, y = np.genfromtxt(os.path.join(DATA_DIR,
                                      "lightcurve_{0}.txt".format(sid))).T
    return x - x[0], y - 1


def make_plots(id, RESULTS_DIR="/Users/ruthangus/projects/GProtation/code/" \
               "results_acfprior_03_10"):
    """
    Make a plot of the fit to the light curve and the posteriors.
    """

    """ load lc """
    x, y = load_suzanne_lcs(id)
    yerr = np.ones(len(y))*1e-5
    m = x < 100
    x, y, yerr = x[m], y[m], yerr[m]

    """ load posteriors """
    fn = os.path.join(RESULTS_DIR, "{}.h5".format(id))
    df = pd.read_hdf(fn, key="samples")

    """ find medians """
    theta = [np.median(df.iloc[:, i]) for i in range(5)]

    """ fit GP """
    print(np.exp(theta[-1]), "period")
    k = theta[0] * ExpSquaredKernel(theta[1]) \
        * ExpSine2Kernel(theta[2], theta[4]) + WhiteKernel(theta[3])
    gp = george.GP(k, solver=george.HODLRSolver)
    gp.compute(x-x[0], yerr)
    xs = np.linspace((x-x[0])[0], (x-x[0])[-1], 1000)
    mu, cov = gp.predict(y, xs)

    """ plot fit """
    plt.clf()
    plt.plot(x, y, "k.")
    plt.plot(xs, mu)
    plt.xlim(0, 100)
    v = np.std(y)
    plt.ylim(-10*v, 10*v)
    plt.savefig("{}_fit".format(id))

    """ plot corner """
    fig = corner.corner(df)
    fig.savefig("{}_corner".format(id))

if __name__ == "__main__":
    make_plots(337)
