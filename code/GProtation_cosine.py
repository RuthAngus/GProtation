from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import CosineKernel, ExpSquaredKernel
import glob
import emcee
import triangle
import h5py
import subprocess
from plotstuff import params, colours
reb = params()
cols = colours()

plotpar = {'axes.labelsize': 25,
           'font.size': 25,
           'legend.fontsize': 25,
           'xtick.labelsize': 25,
           'ytick.labelsize': 25,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def lnprior(theta, plims):
    """
    plims is a tuple, list or array containing the lower and upper limits for
    the rotation period. These are not logarithmic!
    """
    if -20 < theta[0] < 16 and -20 < theta[1] < 20 and -20 < theta[2] < 20 \
    and np.log(plims[0]) < theta[3] < np.log(plims[1]):
        return 0.
    return -np.inf

# lnprob
def lnprob(theta, x, y, yerr, plims):
    return lnlike(theta, x, y, yerr) + lnprior(theta, plims)

# lnlike
def lnlike(theta, x, y, yerr):
    theta = np.exp(theta)
    k = theta[0] * ExpSquaredKernel(theta[1]) \
            * CosineKernel(theta[3])
    gp = george.GP(k)
    try:
        gp.compute(x, np.sqrt(theta[2]+yerr**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    return gp.lnlikelihood(y, quiet=True)

# make various plots
def make_plot(sampler, x, y, yerr, ID, DIR, traces=False):

    nwalkers, nsteps, ndim = np.shape(sampler.chain)
    flat = sampler.flatchain
    result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(flat, [16, 50, 84], axis=0)))
    print(result)
    print("\n", np.exp(np.array(result[-1])), "period (days)", "\n")
    mcmc_result = np.array([i[0] for i in result])
    print(np.exp(mcmc_result))
    np.savetxt("%s/%s_result.txt" % (DIR, ID), result)

    fig_labels = ["A", "l", "s", "P"]

    if traces:
        print("Plotting traces")
        for i in range(ndim):
            plt.clf()
            plt.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3)
            plt.ylabel(fig_labels[i])
            plt.savefig("%s/%s_%s.png" % (DIR, ID, fig_labels[i]))

    print("Making triangle plot")
    fig = triangle.corner(flat, labels=fig_labels)
    fig.savefig("%s/%s_triangle" % (DIR, ID))
    print("%s/%s_triangle.png" % (DIR, ID))

    print("plotting prediction")
    theta = np.exp(np.array(mcmc_result))
    k = theta[0] * ExpSquaredKernel(theta[1]) \
            * CosineKernel(theta[3])
    gp = george.GP(k)
    gp.compute(x, yerr)
    xs = np.linspace(x[0], x[-1], 1000)
    mu, cov = gp.predict(y, xs)
    sig = np.sqrt(np.diag(cov))

    print(sig)

    plt.clf()
    plt.errorbar(x, y, yerr=yerr, alpha=.5, **reb)
    plt.plot(xs, mu, color=cols.lightblue, linewidth=3)
    plt.subplots_adjust(bottom=.2, left=.2, top=.9, right=.9)
    plt.xlabel("$\mathrm{Time~(days)}$")
    plt.ylabel("$\mathrm{Normalised~flux}$")
    plt.savefig("%s/%s_prediction" % (DIR, ID))
    print("%s/%s_prediction.png" % (DIR, ID))

    return result

# take x, y, yerr and initial guess and do MCMC
def MCMC(theta_init, x, y, yerr, plims, burnin, run, ID, DIR, logsamp=True):

    ndim, nwalkers = len(theta_init), 32
    p0 = [theta_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (x, y, yerr, plims)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)

    print("burning in...")
    p0, lp, state = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    print("production run...")
    p0, lp, state = sampler.run_mcmc(p0, run)

    # save samples
    f = h5py.File("%s/%s_samples.h5" % (DIR, ID), "w")
    data = f.create_dataset("samples", np.shape(sampler.chain))
    data[:, :] = np.array(sampler.chain)
    f.close()
    return sampler
