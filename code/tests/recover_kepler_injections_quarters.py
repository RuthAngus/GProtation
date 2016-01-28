import numpy as np
import matplotlib.pyplot as plt
from Kepler_ACF import corr_run
import h5py
from gatspy.periodic import LombScargle
import sys
from Kepler_ACF import corr_run
from multiprocessing import Pool
import glob
from measure_GP_rotation import bin_data
from recover_kepler_injections import recover_injections

def fit_quarters(x, y, yerr, id, p_init, plims, DIR, burnin=500, run=1500,
        npts=48, cutoff=100, sine_kernel=False, acf=False, runMCMC=True,
        plot=False):
    """
    takes x, y, yerr and initial guesses and priors for period and does
    the full GP MCMC.
    Tuning parameters include cutoff (number of days), npts (number of points
    per bin).
    DIR is where to save output
    Also, divide data into quarters

    Changes
    Subsampling, not binning.
    Splitting into qs.
    Running MCMC from GProtation_quarters.
    """

#     xb, yb, yerrb = bin_data(x, y, yerr, npts)  # bin data
    xsub, ysub, yerrsub = x[::npts], y[::npts], yerr[::npts]  # subsample data
    xb, yb, yerrb = split_into_quarters(xsub, ysub, yerrsub)
    nquarters = len(newx)

    # measure ACF period
    if acf:
        corr_run(x, y, yerr, id, "/Users/angusr/Python/GProtation/code")

    if sine_kernel:
        print "sine kernel"
        As = np.ones(nquarters) * np.exp(-5)
        sigmas = np.ones(nquarters) * np.exp(-16)
        theta_init = [As, np.exp(7), np.exp(.6), sigmas, p_init]
        theta_init = [i for j in theta_init for i in j]
        print theta_init
        from GProtation_quarters import MCMC, make_plot
    else:
        print "cosine kernel"
        theta_init = [1e-2, 1., 1e-2, p_init]
        print "theta_init = ", np.log(theta_init)
        from GProtation_cosine import MCMC, make_plot

    theta_init = np.log(theta_init)

    print theta_init
    if runMCMC:
        sampler = MCMC(theta_init, xb, yb, yerrb, plims, burnin, run, id, DIR)

    # make various plots
    if plot:
        with h5py.File("%s/%s_samples.h5" % (DIR, id), "r") as f:
            samples = f["samples"][...]
        mcmc_result = make_plot(samples, xb, yb, yerrb, x, y, yerr,
                                str(int(id)).zfill(4), DIR, traces=True,
                                tri=True, prediction=True)

def acf_pgram_GP_noisy(id):
    """
    Run acf, pgram and MCMC recovery on noisy simulations.
    """
    # run full MCMC recovery
    id = str(int(id)).zfill(4)
    path = "simulations/kepler_injections"
    x, y, yerr = np.genfromtxt("{0}/{1}.txt".format(path, id)).T  # load data
    periodograms(id, x, y, yerr, path, plot=True)  # pgram
    my_acf(id, x, y, yerr, path, plot=True, amy=True)  # acf
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
    my_acf(id, x, y, yerr, path, plot=True, amy=True)  # acf
    burnin, run = 5000, 10000
    recover_injections(id, x, y, yerr, path, burnin, run, runMCMC=True,
                       plot=True)  # MCMC

def acf_pgram_GP_quarters(id):
    """
    Run acf, pgram and MCMC recovery on noisy simulations with each quarter
    modelled separately.
    """
    # run full MCMC recovery
    id = str(int(id)).zfill(4)
    path = "results"
    x, y, yerr = np.genfromtxt("{0}/{1}.txt".format(path, id)).T  # load data
    _, qs = np.genfromtxt("quarters.txt", skip_header=1).T
#     periodograms(id, x, y, yerr, path, plot=True)  # pgram
#     my_acf(id, x, y, yerr, path, plot=True, amy=True)  # acf
    burnin, run = 5000, 10000
    recover_injections_quarters(id, x, y, yerr, path, burnin, run,
                                runMCMC=True, plot=True)  # MCMC

def split_into_quarters(x, y, yerr):
    _, qs = np.genfromtxt("quarters.txt", skip_header=1).T
    qs -= qs[0]
    newx, newy, newyerr = [], [], []
    for i in range(len(qs)-1):
        m = (qs[i] < x) * (x < qs[i+1])
        if len(x[m]):  # only if that quarter exists
            newx.append(x[m])
            newy.append(y[m])
            newyerr.append(yerr[m])
    m = (x > qs[-1])
    newx.append(x[m])
    newy.append(y[m])
    newyerr.append(yerr[m])
    return newx, newy, newyerr

if __name__ == "__main__":

    acf_pgram_GP_quarters(0)

#     # noisy simulations
#     N = 2
#     ids = range(N)
#     ids = [str(int(i)).zfill(4) for i in ids]
#     pool = Pool()
#     pool.map(acf_pgram_GP_quarters, ids)
