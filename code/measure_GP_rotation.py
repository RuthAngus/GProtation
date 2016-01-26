import numpy as np
import matplotlib.pyplot as plt
import kplr
client = kplr.API()
import pyfits
import glob
from Kepler_ACF import corr_run
import h5py

def get_data(id):
    fnames = glob.glob("/Users/angusr/.kplr/data/lightcurves/%s/*" %
                       str(int(id)).zfill(9))
    hdulist = pyfits.open(fnames[0])
    t = hdulist[1].data
    time = t["TIME"]
    flux = t["PDCSAP_FLUX"]
    flux_err = t["PDCSAP_FLUX_ERR"]
    q = t["SAP_QUALITY"]
    m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err) * (q==0)
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
                (q==0)
        x = np.concatenate((x, time[m]))
        med = np.median(flux[m])
        y = np.concatenate((y, flux[m]/med - 1))
        yerr = np.concatenate((yerr, flux_err[m]/med))
    return x, y, yerr

def bin_data(x, y, yerr, npts):
    mod, nbins = len(x) % npts, len(x) / npts
    if mod != 0:
        x, y, yerr = x[:-mod], y[:-mod], yerr[:-mod]
    xb, yb, yerrb = [np.zeros(nbins) for i in range(3)]
    for i in range(npts):
        xb += x[::npts]
        yb += y[::npts]
        yerrb += yerr[::npts]**2
        x, y, yerr = x[1:], y[1:], yerr[1:]
    return xb/npts, yb/npts, yerrb**.5/npts

def fit(x, y, yerr, id, p_init, plims, burnin=500, run=1500, npts=48,
        cutoff=100, sine_kernel=False, acf=False, runMCMC=True, plot=False):
    """
    takes x, y, yerr and initial guesses and priors for period and does
    the full GP MCMC.
    Tuning parameters include cutoff (number of days), npts (number of points
    per bin).
    """

    # measure ACF period
    if acf:
        corr_run(x, y, yerr, id, "/Users/angusr/Python/GProtation/code")

    if sine_kernel:
        print "sine kernel"
        theta_init = [np.exp(-5), np.exp(7), np.exp(.6), np.exp(-16), p_init]
        print theta_init
        from GProtation import MCMC, make_plot
    else:
        print "cosine kernel"
        theta_init = [1e-2, 1., 1e-2, p_init]
        print "theta_init = ", np.log(theta_init)
        from GProtation_cosine import MCMC, make_plot

    xb, yb, yerrb = bin_data(x, y, yerr, npts)  # bin data
    m = xb < cutoff  # truncate

    theta_init = np.log(theta_init)
    DIR = "cosine"
    if sine_kernel:
        DIR = "sine"

    print theta_init
    if runMCMC:
        sampler = MCMC(theta_init, xb[m], yb[m], yerrb[m], plims, burnin, run,
                       id, DIR)

    # make various plots
    if plot:
        with h5py.File("%s/%s_samples.h5" % (DIR, str(int(id)).zfill(4)),
                       "r") as f:
            samples = f["samples"][...]
        m2 = x < cutoff
        mcmc_result = make_plot(samples, xb[m], yb[m], yerrb[m], x[m2], y[m2],
                                yerr[m2], id, DIR, traces=True, tri=True,
                                prediction=True)
