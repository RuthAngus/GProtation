import numpy as np
import matplotlib.pyplot as plt
from measure_GP_rotation import fit
from Kepler_ACF import corr_run
import h5py
from plotstuff import params, colours
reb = params()
cols = colours()
from gatspy.periodic import LombScargle
import sys
from simple_acf import simple_acf, make_plot
from Kepler_ACF import corr_run
from multiprocessing import Pool
import glob
from kepler_data import load_kepler_data

plotpar = {'axes.labelsize': 22,
           'font.size': 22,
           'legend.fontsize': 22,
           'xtick.labelsize': 22,
           'ytick.labelsize': 22,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def my_acf(id, x, y, yerr, fn, plot=False, amy=False):
    """
    takes id of the star, returns an array of period measurements and saves the
    results.
    (the files are saved in a directory that is a global variable).
    """
    if amy:
        period, period_err = corr_run(x, y, yerr, id, fn, saveplot=plot)
    else:
        period, acf, lags = simple_acf(x, y)
        np.savetxt("{0}/{1}_result.txt".format(fn, id, period))
        if plot:
            make_plot(acf, lags, id, fn)
    return period

def periodograms(id, x, y, yerr, fn, plot=False, savepgram=True):
    """
    takes id of the star, returns an array of period measurements and saves the
    results.
    (the files are saved in a directory that is a global variable).
    """
    # initialise with acf
    try:
        p_init, _ = np.genfromtxt("{0}/{1}_result.txt".format(fn, id))
    except:
        corr_run(x, y, yerr, id, fn, saveplot=False)
        p_init, _ = np.genfromtxt("{0}/{1}_result.txt".format(fn, id))
    print "acf period, err = ", p_init

    ps = np.linspace(p_init*.1, p_init*4, 1000)
    model = LombScargle().fit(x, y, yerr)
    pgram = model.periodogram(ps)

    # find peaks
    peaks = np.array([i for i in range(1, len(ps)-1) if pgram[i-1] <
                     pgram[i] and pgram[i+1] < pgram[i]])
    period = ps[pgram==max(pgram[peaks])][0]
    print "pgram period = ", period

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

def recover_injections(id, x, y, yerr, fn, bi, ru, runMCMC=True, plot=False):
    """
    run MCMC on each star, initialising with the ACF period
    Next you could try running for longer, using more of the lightcurve,
    subsampling less, etc
    takes id, x, y, yerr
    fn = path to where the files are saved and where files are loaded
    bi = number of burn in steps
    ru = number of run steps
    """

    # initialise with acf
    try:
        p_init = np.genfromtxt("{0}/{1}_result.txt".format(fn, id))
    except:
        corr_run(x, y, yerr, id, fn, saveplot=plot)
        p_init = np.genfromtxt("{0}/{1}_result.txt".format(fn, id))
    print "acf period, err = ", p_init

    # run MCMC
    plims = [p_init[0] - .99*p_init[0], p_init[0] + 3*p_init[0]]
    npts = int(p_init[0] / 10. * 48)  # 10 points per period
    cutoff = 10 * p_init[0]
    ppd = 48. / npts
    ppp = ppd * p_init[0]
    print("npts =", npts, "cutoff =", cutoff, "points per day =", ppd,
          "points per period =", ppp)
    fit(x, y, yerr, str(int(id)).zfill(4), p_init[0], np.log(plims), fn,
            burnin=bi, run=ru, npts=npts, cutoff=cutoff,
            sine_kernel=True, acf=False, runMCMC=runMCMC, plot=plot)

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

def acf_pgram_GP_sim(id):
    """
    Run acf, pgram and MCMC recovery on noise-free simulations
    """
    id = str(int(id)).zfill(4)
    path = "simulations/noise-free"
    x, y, yerr = np.genfromtxt("{0}/{1}.txt".format(path, id)).T  # load
    periodograms(id, x, y, yerr, path, plot=True)  # pgram
    my_acf(id, x, y, yerr, path, plot=True, amy=True)  # acf
    burnin, run = 5000, 10000
    recover_injections(id, x, y, yerr, path, burnin, run, runMCMC=True,
                       plot=True)  # MCMC

if __name__ == "__main__":

#     # noise-free simulations
#     N = 2
#     ids = range(N)
#     ids = [str(int(i)).zfill(4) for i in ids]
#     pool = Pool()
#     pool.map(acf_pgram_GP_sim, ids)
#     acf_pgram_GP_sim(0)

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
