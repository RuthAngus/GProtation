import numpy as np
import matplotlib.pyplot as plt
from GProtation import MCMC
from Kepler_ACF import corr_run
import h5py
from gatspy.periodic import LombScargle
import sys
import os
import time
from GProtation import make_plot, lnprob

plotpar = {'axes.labelsize': 22,
           'font.size': 22,
           'legend.fontsize': 22,
           'xtick.labelsize': 22,
           'ytick.labelsize': 22,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def my_acf(id):
    id = str(int(id)).zfill(4)
    x, y = np.genfromtxt("simulations/lightcurve_{0}.txt".format(id)).T
    yerr = np.ones_like(y) * 1e-5
    period, p_err = corr_run(x, y, yerr, id, "simulations")
    periods.append(period)
    np.savetxt("simulations/{0}_acf_result.txt".format(id),
               np.array([period, p_err]).T)

def periodograms(id, plot=False, savepgram=True):

    id = str(int(i)).zfill(4)

    # load simulated data
    x, y = np.genfromtxt("simulations/lightcurve_{0}.txt".format(id)).T
    yerr = np.ones_like(y) * 1e-5

    # initialise with acf
    fname = "simulations/{0}_acf_result.txt".format(id)
    if os.path.exists(fname):
        p_init, err = np.genfromtxt(fname).T
    else:
        p_init, err = corr_run(x, y, yerr, id, "simulations")
        np.savetxt("simulations/{0}_acf_result.txt".format(id),
                   np.array([period, p_err]).T)
    print "acf period, err = ", p_init, err

    ps = np.linspace(p_init*.1, p_init*4, 1000)
    model = LombScargle().fit(x, y, yerr)
    pgram = model.periodogram(ps)

    # find peaks
    peaks = np.array([i for i in range(1, len(ps)-1) if pgram[i-1] <
                     pgram[i] and pgram[i+1] < pgram[i]])
    period = ps[pgram==max(pgram[peaks])][0]
    periods.append(period)
    print "pgram period = ", period

    plt.clf()
    plt.plot(ps, pgram)
    plt.axvline(period, color="r")
    plt.savefig("simulations/{0}_pgram".format(id))

    data = np.array([period, period])
    np.savetxt("simulations/periodogram_results.txt", data.T)

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
        print(fname)
        assert 0
        p_init, err = corr_run(x, y, yerr, id, "simulations")
        np.savetxt("simulations/{0}_acf_result.txt".format(id),
                   np.array([p_init, err]).T)
    print "acf period, err = ", p_init, err

    # Format data
    sub = int(p_init / npts * 48)  # 10 points per period
    ppd = 48. / sub
    ppp = ppd * p_init[0]
    print("sub = ", sub, "points per day =", ppd, "points per period =",
          ppp)
    xsub, ysub, yerrsub = x[::sub], y[::sub], yerr[::sub]

    # prep MCMC
    plims = np.log([p_init*.7, p_init*1.5])
    npts = int(p_init / 20. * 48)  # 10 points per period
    cutoff = 10 * p_init
    theta_init = np.log([np.exp(-5), np.exp(7), np.exp(.6), np.exp(-16),
                        p_init])
    burnin, run, nwalkers = 5, 50, 12

    # time the lhf call
    start = time.time()
    print("lnprob = ", lnprob(theta_init, xb, yb, yerrb, plims))
    end = time.time()
    tm = end - start
    print("1 lhf call takes ", tm, "seconds")
    print("burn in will take", tm * nwalkers * burnin, "s")
    print("run will take", tm * nwalkers * run, "s")
    print("total = ", (tm * nwalkers * run + tm * nwalkers * burnin)/60, \
          "mins")

    MCMC(theta_init, xsub, ysub, yerrsub, plims, burnin, run, id,
         "simulations", nwalkers)

if __name__ == "__main__":

    # run full MCMC recovery
    recover_injections(2)
