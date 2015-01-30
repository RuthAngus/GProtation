from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import glob
from ACF import load_fits
import emcee
import triangle
import h5py
import subprocess

from params import plot_params
reb = plot_params()
from colours import plot_colours
cols = plot_colours()

# take x, y, yerr, initial guess and fit an initial GP
# prompt for initial guess
# return GP params
def basic(x, y, yerr, p_init, ID, DIR):

    # initialise
    theta_init = [1, 1, 1, p_init, 1.]
    k = theta_init[0] * ExpSquaredKernel(theta_init[1]) \
            * ExpSine2Kernel(theta_init[2], theta_init[3])
    gp = george.GP(k)
    gp.compute(x, np.sqrt(theta[4]**2+yerr**2))

    # optimise
    theta = gp.optimize(x, y, np.sqrt(theta[4]**2+yerr**2))[0]

    # predict
    xs = np.linspace(x[0], x[-1], 1000)
    mu, cov = gp.predict(y, xs)

    plt.clf()
    plt.errorbar(x, y, yerr=np.sqrt(theta[4]**2+yerr**2), **reb)
    plt.plot(xs, mu, color=cols.blue)
    plt.savefig("%s/%s_predict" % (DIR, ID))

    return theta

# model
def model(theta, x, y, yerr):
    k = theta[0] * ExpSquaredKernel(theta[1]) \
            * ExpSine2Kernel(theta[2], theta[3])
    gp = george.GP(k)
    gp.compute(x, yerr)

    # predict
    mu, cov = gp.predict(y, x)
    return mu, cov

# lnprior
def lnprior(theta):
    if 1e-10 < theta[0] < 1e10 and 1e-10 < theta[1] < 1e10 and \
            1e-10 < theta[2] < 1e10 and 0. < theta[3] < 100. and \
            1e-10 < theta[4] < 1e10:
        return 0.0
    return -np.inf

# lnprob
def lnprob(theta, x, y, yerr):
    return lnlike(theta, x, y, yerr) + lnprior(theta)

# lnlike
def lnlike(theta, x, y, yerr):
    k = theta[0] * ExpSquaredKernel(theta[1]) \
            * ExpSine2Kernel(theta[2], theta[3])
    gp = george.GP(k)
    try:
        gp.compute(x, np.sqrt(theta[4]+yerr**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    return gp.lnlikelihood(y, quiet=True)

# make various plots
def make_plot(sampler, ID, DIR, traces=False):

    nwalkers, nsteps, ndim = np.shape(sampler.chain)
    flat = sampler.chain[:, 50:, :].reshape((-1, ndim))
    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(flat, [16, 50, 84], axis=0)))
    print(mcmc_result)

    if traces:
        print("Plotting traces")
        for i in range(ndim):
            plt.clf()
            plt.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3)
            plt.savefig("%s/%s.png" % (DIR, i))

    print("Making triangle plot")
    fig_labels = ["$A$", "$l_2$", "$l_1$", "$s$", "$P$", "$wn$"]
    fig = triangle.corner(sampler.flatchain, labels=fig_labels)
    fig.savefig("%s/%s_triangle.png" % (DIR, ID))

# take x, y, yerr and initial guess and do MCMC
def MCMC(theta_init, x, y, yerr, ID, DIR):

    ndim, nwalkers = len(theta_init), 32
    p0 = [theta_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (x, y, yerr)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)

    print("burning in...")
    p0, lp, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    print("production run...")
    p0, lp, state = sampler.run_mcmc(p0, 1000)

    f = h5py.File("%s/%s_samples.h5" % (DIR, ID), "w")
    data = f.create_dataset("samples", np.shape(sampler.chain))
    data[:, :] = np.array(sampler.chain)
    f.close()

    return sampler

def wrapper(x, y, yerr, ID, DIR):

        # subsample
        s = 10
        x, y, yerr = x[::s], y[::s], yerr[::s]

        # median normalise
        med = np.median(y)
        y /= med
        yerr /= med

        # try optimising
        print(ID)
        p_init = float(raw_input("Enter initial guess for rotation period "))
        theta = basic(x, y, yerr, p_init, ID, DIR)
        print(theta)
        choose = raw_input("Use these parameters? y/n ")
        if choose == "y": theta = theta
        else: theta = [1., 1., 1., p_init, 1.]

#         theta = [1., 1., 1., 8., 1.]
        sampler = MCMC(np.array(theta), x, y, yerr, ID, DIR)
        make_plot(sampler, ID, DIR)

def kepler():
    D = "/Users/angusr/angusr/data2/Q15_public"  # data directory
    DIR = "/Users/angusr/Python/GProtation/mcmc"  # results directory
    fnames = glob.glob("%s/kplr0081*" % D)

    for fname in fnames:
        x, y, yerr = load_fits(fname)
        l = np.isfinite(x) * np.isfinite(y) * np.isfinite(yerr)
        x, y, yerr = x[l], y[l], yerr[l]
        kid = fname[42:51]
        wrap(x, y, yerr, fname, DIR)

if __name__ == "__main__":

    D = "/Users/angusr/data/K2/c0corcutlcs"  # data directory
    DIR = "/Users/angusr/Python/GProtation/mcmc"  # results directory
    fnames = glob.glob("%s/ep*.csv" % D)

    for fname in fnames:
        x, y, empty = np.genfromtxt(fname, skip_header=1, delimiter=",").T
        yerr = np.ones_like(y)*1e-5
        ID = fname[36:45]
        wrapper(x, y, yerr, ID, DIR)
