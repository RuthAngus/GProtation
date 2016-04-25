from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from GProtation import make_plot, lnprob
from Kepler_ACF import corr_run
import h5py
from gatspy.periodic import LombScargle
import sys
import os
import time
import emcee

plotpar = {'axes.labelsize': 22,
           'font.size': 22,
           'legend.fontsize': 22,
           'xtick.labelsize': 22,
           'ytick.labelsize': 22,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def recover_injections(id):
    """
    run MCMC on each star, initialising with the ACF period.
    """
    id = str(int(id)).zfill(4)

    # load simulated data
    x, y = np.genfromtxt("simulations/lightcurve_{0}.txt".format(id)).T
    yerr = np.ones_like(y) * 1e-5

    # initialise with acf
    fname = "simulations/{0}_acf_result.txt".format(id)
    if os.path.exists(fname):
        p_init, err = np.genfromtxt(fname)
    else:
        p_init, err = corr_run(x, y, yerr, id, "simulations")
        np.savetxt("simulations/{0}_acf_result.txt".format(id),
                   np.array([p_init, err]).T)
    print("acf period, err = ", p_init, err)

    # Format data
#     p_init = .5
    npts = 10
    sub = int(p_init / npts * 48)  # 10 points per period
    ppd = 48. / sub
    ppp = ppd * p_init
    print("sub = ", sub, "points per day =", ppd, "points per period =",
          ppp)
    xsub, ysub, yerrsub = x[::sub], y[::sub], yerr[::sub]
    c = 100 * p_init  # cutoff
    m = xsub < (xsub[0] + c)
    xb, yb, yerrb = xsub[m], ysub[m], yerrsub[m]

    # plot data
    plt.clf()
    m = x < (xsub[0] + c)
    plt.errorbar(x[m], y[m], yerr=yerr[m], fmt="k.", capsize=0)
    plt.errorbar(xb, yb, yerr=yerrb, fmt="r.", capsize=0)
    plt.savefig("simulations/{0}_sub".format(id))

    # prep MCMC
    plims = np.log([.1*p_init, 1.9*p_init])
    print("total number of points = ", len(xb))
    theta_init = np.log([np.exp(-5), np.exp(7), np.exp(.6), np.exp(-16),
                        p_init])
    burnin, run, nwalkers = 2000, 5000, 12
    ndim = len(theta_init)
    p0 = [theta_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (xb, yb, yerrb, plims)

    # time the lhf call
    start = time.time()
    print("lnprob = ", lnprob(theta_init, xb, yb, yerrb, plims))
    end = time.time()
    tm = end - start
    print("1 lhf call takes ", tm, "seconds")
    print("burn in will take", tm * nwalkers * burnin, "s")
    print("run will take", tm * nwalkers * run, "s")
    print("total = ", (tm * nwalkers * run + tm * nwalkers * burnin)/60,
          "mins")

    # run MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
    print("burning in...")
    start = time.time()
    p0, _, state = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    print("production run...")
    p0, _, state = sampler.run_mcmc(p0, run)
    end = time.time()
    print("actual time = ", end - start)

    # save samples
    f = h5py.File("simulations/%s_samples.h5" % (id), "w")
    data = f.create_dataset("samples", np.shape(sampler.chain))
    data[:, :] = np.array(sampler.chain)
    f.close()

    # make various plots
    with h5py.File("simulations/%s_samples.h5" % (id), "r") as f:
        samples = f["samples"][...]
    mcmc_result = make_plot(samples, xb, yb, yerrb, id, "simulations",
                            traces=True, tri=True, prediction=True)

if __name__ == "__main__":

    # run full MCMC recovery
    recover_injections(24)
