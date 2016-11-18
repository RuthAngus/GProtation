# This script contains the prior, lhf and logprob functions, plus plotting
# routines.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
import glob
import emcee
import corner
import h5py
import subprocess
import scipy.optimize as spo
import time
import os
import pandas as pd
import time


def lnprior(theta, plims):
    """
    plims is a tuple, list or array containing the lower and upper limits for
    the rotation period. These are logarithmic!
    theta = A, l, Gamma, s, P
    """
    if -20 < theta[0] < 20 and theta[4] < theta[1] and -20 < theta[2] < 20 \
    and -20 < theta[3] < 20 and plims[0] < theta[4] < plims[1] \
    and theta[4] < 4.61:
        return 0.
    return -np.inf


def lnGauss(x, mu, sigma):
    return -.5 * ((x - mu)**2/(.5 * sigma**2))


def Glnprior(theta, p_init, p_max):
    """
    theta = A, l, G, sigma, period
    """
    mu = np.array([-13, 6.2, -1.4, -17, p_init])
    sigma = np.array([2.7, 1.5, 1.5, 5, p_init * 2])
#     if np.log(.5) < theta[4] < p_max and 0 < theta[1] \
#         and theta[1]/theta[4] < 10:
    if np.log(.5) < theta[4] < p_max:
        return np.sum(lnGauss(theta, mu, sigma))
    return -np.inf


def lnprob(theta, x, y, yerr, plims):
    prob = lnlike(theta, x, y, yerr) + lnprior(theta, plims)
    return prob, prob


def Glnprob(theta, x, y, yerr, p_init, p_max):
    prob = lnlike(theta, x, y, yerr) + Glnprior(theta, p_init, p_max)
    return prob, prob


def Glnprob_split(theta, x, y, yerr, p_init, p_max):
    prior = Glnprior(theta, p_init, p_max)
    prob = np.sum([lnlike(theta, x[i], y[i], yerr[i]) + prior for i in
                   range(len(x))])
    return prob, prob


def lnlike(theta, x, y, yerr):
    theta = np.exp(theta)
    k = theta[0] * ExpSquaredKernel(theta[1]) \
            * ExpSine2Kernel(theta[2], theta[4]) + WhiteKernel(theta[3])
    gp = george.GP(k, solver=george.HODLRSolver)
    try:
        gp.compute(x, np.sqrt(theta[3]+yerr**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    return gp.lnlikelihood(y, quiet=True)


def neglnlike(theta, x, y, yerr):
    theta = np.exp(theta)
    k = theta[0] * ExpSquaredKernel(theta[1]) \
            * ExpSine2Kernel(theta[2], theta[4])
    gp = george.GP(k)
    try:
        gp.compute(x, np.sqrt(theta[3]+yerr**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    return -gp.lnlikelihood(y, quiet=True)

# make various plots
def make_plot(sampler, xb, yb, yerrb, ID, RESULTS_DIR, trths, traces=False,
              tri=False, prediction=True):

    nwalkers, nsteps, ndims = np.shape(sampler)
    flat = np.reshape(sampler, (nwalkers * nsteps, ndims))
    mcmc_res = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(flat, [16, 50, 84], axis=0)))
    med = np.concatenate([np.array(mcmc_res[i]) for i in
                          range(len(mcmc_res))])
    print("median values = ", med[::3])
    logprob = flat[:, -1]
    ml = logprob == max(logprob)
    maxlike = flat[np.where(ml)[0][0], :][:-1]
    print("max like = ", maxlike)
    print("\n", np.exp(np.array(maxlike[-1])), "period (days)", "\n")
    r = np.concatenate((maxlike, med))

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
            plt.plot(sampler[:, :, i].T, 'k-', alpha=0.3)
            plt.ylabel(fig_labels[i])
            plt.savefig(os.path.join(RESULTS_DIR, "{0}_{1}.png".format(ID,
                        fig_labels[i])))

    if tri:
        DIR = "../code/simulations/kepler_diffrot_full/par/"
        truths = pd.read_csv(os.path.join(DIR, "final_table.txt"),
                             delimiter=" ")
        true_p = np.log(truths.P_MIN.values[truths.N.values ==
                                    int(filter(str.isdigit, ID))][0])
        print("Making triangle plot")
        fig = corner.corner(flat[:, :-1], labels=fig_labels,
                            quantiles=[.16, .5, .84], show_titles=True,
                            truths=trths)
        fig.savefig(os.path.join(RESULTS_DIR, "{0}_triangle".format(ID)))
        print(os.path.join("{0}_triangle.png".format(ID)))

    if prediction:
        if len(xb) > 1:
            x = [i for j in xb for i in j]
            y = [i for j in yb for i in j]
            yerr = [i for j in yerrb for i in j]
        else:
            x, y, yerr = xb, yb, yerrb
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
