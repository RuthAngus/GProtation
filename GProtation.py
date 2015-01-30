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
