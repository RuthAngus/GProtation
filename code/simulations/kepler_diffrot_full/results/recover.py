from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import h5py
from plotstuff import params, colours
reb = params()
cols = colours()
from gatspy.periodic import LombScargle
import sys
from Kepler_ACF import corr_run
from multiprocessing import Pool
import glob
from kepler_data import load_kepler_data
from quarters import split_into_quarters, lnprob_split
from GProtation import make_plot, lnprob
import emcee
import time
from simple_acf import simple_acf


def my_acf(id, x, y, yerr, interval, fn, plot=False, amy=False):
    """
    takes id of the star, returns an array of period measurements and saves the
    results.
    (the files are saved in a directory that is a global variable).
    """
    try:
        period, period_err = \
                np.genfromtxt("{0}/{1}_acfresult.txt".format(fn, id))
    except:
        if amy:
            period, period_err = corr_run(x, y, yerr, id, interval, fn,
                                          saveplot=plot)
        else:
            period, acf_smooth, lags = simple_acf(x, y, interval)
            np.savetxt("{0}/{1}_acfresult.txt".format(fn, id),
                       np.transpose((period, period*.1)))
    return period

def periodograms(id, x, y, yerr, interval, fn, plot=False, savepgram=True):
    """
    takes id of the star, returns an array of period measurements and saves the
    results.
    (the files are saved in a directory that is a global variable).
    """
    # initialise with acf
    try:
        p_init, err = np.genfromtxt("{0}/{1}_acfresult.txt".format(fn, id))
    except:
        corr_run(x, y, yerr, id, interval, fn, saveplot=False)
        p_init, err = np.genfromtxt("{0}/{1}_acfresult.txt".format(fn, id))
    print("acf period, err = ", p_init, err)

    ps = np.linspace(p_init*.1, p_init*4, 1000)
    model = LombScargle().fit(x, y, yerr)
    pgram = model.periodogram(ps)

    # find peaks
    peaks = np.array([i for i in range(1, len(ps)-1) if pgram[i-1] <
                     pgram[i] and pgram[i+1] < pgram[i]])
    period = ps[pgram==max(pgram[peaks])][0]
    print("pgram period = ", period)

    if plot:
        plt.clf()
        plt.plot(ps, pgram)
        plt.axvline(period, color="r")
        plt.savefig("{0}/{1}_pgram".format(fn, str(int(id)).zfill(4)))

    if savepgram:
        np.savetxt("{0}/{1}_pgram.txt".format(fn, str(int(id)).zfill(4)),
                   np.transpose((ps, pgram)))

    np.savetxt("{0}/{1}_pgram_result.txt".format(fn, str(int(id)).zfill(4)),
               np.ones(2).T*period)
    return period

def plot_init(theta_init, x, y, yerr):
    print("plotting inits")
    print(np.exp(theta_init))
    t = np.exp(theta_init)
    k = t[0] * ExpSquaredKernel(t[1]) * ExpSine2Kernel(t[2], t[3])
    gp = george.GP(k)
    gp.compute(x, yerr)
    xs = np.linspace(x[0], x[-1], 1000)
    mu, cov = gp.predict(y, xs)

    plt.clf()
    plt.errorbar(x, y, yerr=yerr, **reb)
    plt.plot(xs, mu, color=cols.blue)

    args = (x, y, yerr)
    results = spo.minimize(neglnlike, theta_init, args=args)
    print("optimisation results = ", results.x)

    r = np.exp(results.x)
    k = r[0] * ExpSquaredKernel(r[1]) * ExpSine2Kernel(r[2], r[3])
    gp = george.GP(k)
    gp.compute(x, yerr)

    mu, cov = gp.predict(y, xs)
    plt.plot(xs, mu, color=cols.pink, alpha=.5)
    plt.savefig("%s/%s_init" % (fn, id))
    print("%s/%s_init.png" % (fn, id))


def recover_injections(id, x, y, yerr, fn, burnin, run, npts=10, nwalkers=32,
                       plot_inits=False, plot=True, quarters=False):
    """
    Take x, y, yerr, calculate ACF period for initialisation and do MCMC.
    npts: number of points per period.
    """

    # initialise with acf
    try:
        p_init = np.genfromtxt("{0}/{1}_acfresult.txt".format(fn, id))
    except:
        corr_run(x, y, yerr, id, fn, saveplot=plot)
        p_init = np.genfromtxt("{0}/{1}_acfresult.txt".format(fn, id))
    print("acf period, err = ", p_init)

    # Format data
    plims = np.log([p_init[0] - .4 * p_init[0], p_init[0] + .4 * p_init[0]])
    sub = int(p_init[0] / npts * 48)  # 10 points per period
    ppd = 48. / sub
    ppp = ppd * p_init[0]
    print("sub = ", sub, "points per day =", ppd, "points per period =", ppp)
    # subsample
    xsub, ysub, yerrsub = x[::sub], y[::sub], yerr[::sub]
    if quarters:
        xb, yb, yerrb = split_into_quarters(xsub, ysub, yerrsub)
    else:
        xb, yb, yerrb = x, y, yerr

#     plt.clf()
#     plt.plot(x, y, "k.")
#     for i in range(len(xb)):
#         plt.plot(xb[i], yb[i], "r.")
#     plt.xlim(min(x), min(x) + 10 * p_init[0])
#     plt.savefig("{0}/{1}_test".format(fn, id))

    # assign theta_init
    theta_init = [np.exp(-5), np.exp(7), np.exp(.6), np.exp(-16), p_init[0]]
    theta_init = np.log(theta_init)
    print("\n", "log(theta_init) = ", theta_init)
    print("theta_init = ", np.exp(theta_init), "\n")

    if plot_inits:  # plot the initialisation
        plot_init(theta_init, x, y, err)

    # set up MCMC
    ndim, nwalkers = len(theta_init), nwalkers
    p0 = [theta_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (xb, yb, yerrb, plims)
    lp = lnprob
    if quarters:  # if fitting each quarter separately, use a different lnprob
        lp = lnprob_split

    # time the lhf call
    start = time.time()
    print("lnprob = ", lp(theta_init, xb, yb, yerrb, plims))
    end = time.time()
    tm = end - start
    print("1 lhf call takes ", tm, "seconds")
    print("burn in will take", tm * nwalkers * burnin, "s")
    print("run will take", tm * nwalkers * run, "s")
    print("total = ", (tm * nwalkers * run + tm * nwalkers * burnin)/60, \
          "mins")

    # run MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, args=args)
    print("burning in...")
    start = time.time()
    p0, lp, state = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    print("production run...")
    p0, lp, state = sampler.run_mcmc(p0, run)
    end = time.time()
    print("actual time = ", end - start)

    # save samples
    f = h5py.File("%s/%s_samples.h5" % (fn, id), "w")
    data = f.create_dataset("samples", np.shape(sampler.chain))
    data[:, :] = np.array(sampler.chain)
    f.close()

    # make various plots
    if plot:
        with h5py.File("%s/%s_samples.h5" % (fn, id), "r") as f:
            samples = f["samples"][...]
        mcmc_result = make_plot(samples, xsub, ysub, yerrsub, id, fn,
                                traces=True, tri=True, prediction=True)


def acf_pgram_GP_noisy(id):
    """
    Run acf, pgram and MCMC recovery on noisy simulations.
    """
    # run full MCMC recovery
    id = str(int(id)).zfill(4)
    path = "simulations/kepler_injections"
    x, y, yerr = np.genfromtxt("{0}/{1}.txt".format(path, id)).T  # load data
    periodograms(id, x, y, yerr, path, plot=True)  # pgram
    my_acf(id, x, y, yerr, path, plot=True)  # acf
    burnin, run = 5000, 10000
    recover_injections(id, x, y, yerr, path, burnin, run, runMCMC=True,
                       plot=True)  # MCMC

def acf_pgram_GP(id):
    """
    Run acf, pgram and MCMC recovery on real kepler light curves.
    """
    id = str(int(id)).zfill(9)
    p = "/home/angusr/.kplr/data/lightcurves"
    fnames = np.sort(glob.glob("{0}/{1}/*llc.fits".format(p, id)))
    x, y, yerr = load_kepler_data(fnames)  # load data
    path = "real_lcs"
    periodograms(id, x, y, yerr, path, plot=True)  # pgram
    my_acf(id, x, y, yerr, path, plot=True, amy=False)  # acf
    burnin, run = 5000, 10000
    recover_injections(id, x, y, yerr, path, burnin, run, runMCMC=True,
                       plot=True)  # MCMC

def acf_pgram_GP_sim(id):
    """
    Run acf, pgram and MCMC recovery on noise-free simulations
    """
    id = str(int(id)).zfill(4)
    path = "../../noise-free"
    x, y = np.genfromtxt("{0}/{1}.txt".format(path, id)).T  # load
    yerr = np.ones_like(y) * 1e-8
#     periodograms(id, x, y, yerr, path, plot=True)  # pgram
    my_acf(id, x, y, yerr, path, plot=True, amy=True)  # acf
#     burnin, run = 5000, 10000
#     recover_injections(id, x, y, yerr, path, burnin, run, runMCMC=True,
#                        plot=True)  # MCMC

def acf_pgram_GP_suz(id):
    """
    Run acf, pgram and MCMC recovery on Suzanne's simulations
    """
    noise_free = False
    id = str(int(id)).zfill(4)
    if noise_free:
        path = "noise-free"  # where to save results
        x, y = np.genfromtxt("../noise_free/lightcurve_{0}.txt".format(id)).T
        interval = (x[1] - x[0])
    else:
        path = "noisy"  # where to save results
        x, y = np.genfromtxt("../final/lightcurve_{0}.txt".format(id)).T
        interval = 0.02043365
    yerr = np.ones_like(y) * 1e-5
    periodograms(id, x, y, yerr, interval, path, plot=True)  # pgram
    my_acf(id, x, y, yerr, interval, path, plot=True, amy=False)  # acf
    burnin, run, npts = 1000, 15000, 20  # MCMC
    recover_injections(id, x, y, yerr, path, burnin, run, npts, nwalkers=12,
                       plot=True, quarters=True)

if __name__ == "__main__":

    # Suzanne's noise-free simulations
    data = np.genfromtxt("../par/final_table.txt", skip_header=1).T
    m = data[13] == 0  # just the stars without diffrot
    ids = data[0][m]
    pool = Pool()
    pool.map(acf_pgram_GP_suz, ids)
#     acf_pgram_GP_suz(0)

#     # my noise-free simulations
#     N = 60
#     ids = range(N)
#     pool = Pool()
#     pool.map(acf_pgram_GP_sim, ids)

#     # noisy simulations
#     N = 2
#     ids = range(N)
#     ids = [str(int(i)).zfill(4) for i in ids]
#     pool = Pool()
#     pool.map(acf_pgram_GP_noisy, ids)

#     # real lcs
#     data = np.genfromtxt("data/garcia.txt", skip_header=1).T
#     kids = [str(int(i)).zfill(9) for i in data[0]]
#     pool = Pool()
#     pool.map(acf_pgram_GP, kids)
