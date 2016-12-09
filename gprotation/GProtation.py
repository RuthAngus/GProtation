# This script contains the prior, lhf and logprob functions, plus plotting
# routines.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
import emcee3
import corner


class MyModel(object):
    """
    Model for emcee3
    """
    def __init__(self, x, y, yerr, p_init, p_max):

        self.p_init = p_init
        self.p_max = p_max
        self.x = x
        self.y = y
        self.yerr = yerr

    def lnGauss(self, t, mu, sigma):
        return -.5 * ((t - mu)**2/(.5 * sigma**2))

    def Glnprior(self, theta):
        """
        theta = A, l, G, sigma, period
        """
        mu = np.array([-13, 6.2, -1.4, -17, self.p_init])
        sigma = np.array([2.7, 1.5, 1.5, 5, self.p_init * 2])
        if np.log(.5) < theta[4] < self.p_max and 0 < theta[1]:
            return np.sum(self.lnGauss(theta, mu, sigma))
        return -np.inf

    def lnlike_split(self, theta):
        return np.sum([self.lnlike(theta, self.x[i], self.y[i], self.yerr[i])
                       for i in range(len(self.x))])

    def lnlike(self, theta, xi, yi, yerri):
        theta = np.exp(theta)
        k = theta[0] * ExpSquaredKernel(theta[1]) \
            * ExpSine2Kernel(theta[2], theta[4]) + WhiteKernel(theta[3])
        gp = george.GP(k, solver=george.HODLRSolver)
        try:
            gp.compute(xi, np.sqrt(theta[3]+yerri**2))
        except (ValueError, np.linalg.LinAlgError):
            return 10e25
        return gp.lnlikelihood(yi, quiet=True)


def make_plot(sampler, xb, yb, yerrb, ID, RESULTS_DIR, trths, traces=False,
              tri=False, prediction=True):

    _, ndims = np.shape(sampler.get_coords(flat=True))
    flat = sampler.get_coords(flat=True)
    logprob = sampler.get_log_probability(flat=True)
    mcmc_res = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                        zip(*np.percentile(flat, [16, 50, 84], axis=0))))
    med = np.concatenate([np.array(mcmc_res[i]) for i in
                          range(len(mcmc_res))])
    print("median values = ", med[::3])
    ml = logprob == max(logprob)
    maxlike = flat[np.where(ml)[0][0], :]
    print("max like = ", maxlike)
    print("\n", np.exp(np.array(maxlike[-1])), "period (days)", "\n")
    r = np.concatenate((maxlike, med))

    # calculate autocorrelation times
    try:
        acorr_t = emcee3.autocorr.integrated_time(flat, c=1)
    except:
        acorr_t = emcee3.autocorr.integrated_time(flat)

    # save data
    df = pd.DataFrame({"N": [ID], "A_max": [r[0]], "l_max": [r[1]],
                       "gamma_max": [r[2]], "period_max": [r[3]],
                       "sigma_max": [r[4]], "A": [r[5]], "A_errp": [r[6]],
                       "A_errm": [r[7]], "l": [r[8]], "l_errp": [r[9]],
                       "l_errm": [r[10]], "gamma": [r[11]],
                       "gamma_errp": [r[12]], "gamma_errm": [r[13]],
                       "sigma": [r[14]], "sigma_errp": [r[15]],
                       "sigma_errm": [r[16]], "period": [r[17]],
                       "period_errp": [r[18]], "period_errm": [r[19]],
                       "acorr_A": acorr_t[0], "acorr_l": acorr_t[1],
                       "acorr_gamma": acorr_t[2], "acorr_sigma": acorr_t[3],
                       "acorr_period": acorr_t[4]})
    df.to_csv(os.path.join(RESULTS_DIR, "{0}_mcmc_results.txt".format(ID)))

    fig_labels = ["ln(A)", "ln(l)", "ln(G)", "ln(s)", "ln(P)", "lnprob"]

    if traces:
        print("Plotting traces")
        for i in range(ndims):
            plt.clf()
            plt.plot(sampler.get_coords()[:, :, i], 'k-', alpha=0.3)
            plt.ylabel(fig_labels[i])
            plt.savefig(os.path.join(RESULTS_DIR, "{0}_{1}.png".format(ID,
                        fig_labels[i])))

    if tri:
        print("Making triangle plot")
        fig = corner.corner(flat, labels=fig_labels, quantiles=[.16, .5, .84],
                            show_titles=True)
        fig.savefig(os.path.join(RESULTS_DIR, "{0}_triangle".format(ID)))
        print(os.path.join("{0}_triangle.png".format(ID)))

    if prediction:
        if len(xb) > 1:  # if the data is a list of lists.
            try:
                x = [i for j in xb for i in j]
                y = [i for j in yb for i in j]
                yerr = [i for j in yerrb for i in j]
            except:  # if the data are just a single list.
                TypeError
                x, y, yerr = xb, yb, yerrb
        else:  # if the data is a list of a single list.
            x, y, yerr = xb[0], yb[0], yerrb[0]
        print("plotting prediction")
        theta = np.exp(np.array(maxlike))
        k = theta[0] * ExpSquaredKernel(theta[1]) \
            * ExpSine2Kernel(theta[2], theta[4]) + WhiteKernel(theta[3])
        gp = george.GP(k, solver=george.HODLRSolver)
        gp.compute(x-x[0], yerr)
        xs = np.linspace((x-x[0])[0], (x-x[0])[-1], 1000)
        mu, cov = gp.predict(y, xs)
        plt.clf()
        plt.errorbar(x-x[0], y, yerr=yerr, fmt="k.", capsize=0)
        plt.xlabel("Time (days)")
        plt.ylabel("Normalised Flux")
        plt.plot(xs, mu, color='#0066CC')
        plt.xlim(min(x-x[0]), max(x-x[0]))
        plt.savefig(os.path.join(RESULTS_DIR, "{0}_prediction".format(ID)))
        print(os.path.join(RESULTS_DIR, "{0}_prediction.png".format(ID)))
    return r
