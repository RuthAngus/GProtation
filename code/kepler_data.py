from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pyfits
import glob
from simple_acf import simple_acf
from measure_GP_rotation import bin_data
import h5py
import time
from GProtation import lnprob


def load_kepler_data(fnames):
    hdulist = pyfits.open(fnames[0])
    t = hdulist[1].data
    time = t["TIME"]
    flux = t["PDCSAP_FLUX"]
    flux_err = t["PDCSAP_FLUX_ERR"]
    q = t["SAP_QUALITY"]
    m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err) * \
            (q == 0)
    x = time[m]
    med = np.median(flux[m])
    y = flux[m]/med - 1
    yerr = flux_err[m]/med
    for fname in fnames[1:]:
       hdulist = pyfits.open(fname)
       t = hdulist[1].data
       time = t["TIME"]
       flux = t["PDCSAP_FLUX"]
       flux_err = t["PDCSAP_FLUX_ERR"]
       q = t["SAP_QUALITY"]
       m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err) * \
               (q == 0)
       x = np.concatenate((x, time[m]))
       med = np.median(flux[m])
       y = np.concatenate((y, flux[m]/med - 1))
       yerr = np.concatenate((yerr, flux_err[m]/med))
    return x, y, yerr


def fit(x, y, yerr, id, p_init, plims, DIR, burnin=500, run=1500, npts=48,
        cutoff=1000, sine_kernel=True, runMCMC=True, plot=False):
    """
    takes x, y, yerr and initial guesses and priors for period and does the
    the GP MCMC.
    id: the kepler id
    p_init: period initial guess
    plims: tuple, upper and lower limit for the prior
    DIR: the directory to save the results
    """
    if sine_kernel:
        print("sine kernel")
        theta_init = [np.exp(-5), np.exp(7), np.exp(.6), np.exp(-16), p_init]
        print("theta init = ", theta_init)
        from GProtation import MCMC, make_plot
    else:
        print("cosine kernel")
        theta_init = [1e-2, 1., 1e-2, p_init]
        print("theta init = ", theta_init)
        from GProtation_cosine import MCMC, make_plot

    xb, yb, yerrb = bin_data(x, y, yerr, npts) # bin data
    m = cutoff  # truncate data
    xb, yb, yerrb = xb[:m], yb[:m], yerrb[:m]

    # plot data
    plt.clf()
    plt.plot(x, y, "k.")
    plt.savefig("test")

    theta_init = np.log(theta_init)

    start = time.time()
    lnprob(theta_init, xb, yb, yerrb, plims)
    end = time.time()
    print("1 likelihood call takes", end-start, "seconds")

    if runMCMC:
        sampler = MCMC(theta_init, xb, yb, yerrb, plims, burnin, run,
                       id, DIR, nwalkers=12)

    # make various plots
    if plot:
        with h5py.File("{0}/{1}_samples.h5".format(DIR, str(int(id).zfill(4))),
                       "r") as f:
            samples = f["samples"][...]
        m = x < cutoff
        mcmc_result = make_plot(samples, x[m], y[m], yerr[m], id, DIR,
                                traces=False, triangle=False, prediction=True)


if __name__ == "__main__":

    # load Kepler IDs
    data = np.genfromtxt("data/garcia.txt", skip_header=1).T
    kids = data[0]
    id = kids[0]

    # load the first light curve
    p = "/home/angusr/.kplr/data/lightcurves"
    path = "{0}/{1}/*fits".format(p, str(int(id)).zfill(9))
    fnames = glob.glob(path)
    x, y, yerr = load_kepler_data(fnames)

    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(x, y, "k.")

    # calculate acf
    period, acf, lags = simple_acf(x, y)

    plt.subplot(2, 1, 2)
    plt.axvline(period, color="r")
    plt.plot(lags, acf)
    plt.savefig("test")

    # run MCMC
    p_init = period
    plims = (period - .2*period, period + .2*period)
    DIR = "results"

    fit(x, y, yerr, id, p_init, plims, DIR, burnin=2, run=60, plot=True)
