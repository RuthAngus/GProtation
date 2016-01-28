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
        np.savetxt("{0}/{1}_result.txt".format(fn, str(id).zfill(4)), period)
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
        p_init, _ = np.genfromtxt("{0}/{1}_result.txt".format(fn,
                                                str(int(id)).zfill(4)))
    except:
        corr_run(x, y, yerr, int(id), fn, saveplot=False)
        p_init, _ = np.genfromtxt("{0}/{1}_result.txt".format(fn, int(id)))
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

def collate(N):
    """
    assemble all the period measurements here
    """
    ids, true_p, true_a = \
            np.genfromtxt("{0}/true_periods_amps.txt".format(fn)).T
    acf_periods, aerrs, GP_periods, gerrp, gerrm = [], [], [], [], []
    p_periods, pids = [], []  # periodogram results. pids is just a place hold
    my_p = []
    gp_samps = np.zeros((len(ids), 50))
    for i, id in enumerate(ids[:N]):

        # load ACF results
        try:
            acfdata = np.genfromtxt("{0}/{1}_result.txt".format(fn, int(id))).T
            acf_periods.append(acfdata[0])
            aerrs.append(acfdata[1])
        except:
            acf_periods.append(0)
            aerrs.append(0)

        try:
            # load GP results
            with h5py.File("sine/%s_samples.h5" % str(int(id)).zfill(4),
                           "r") as f:
                samples = f["samples"][...]
            nwalkers, nsteps, ndim = np.shape(samples)
            flat = np.reshape(samples, (nwalkers*nsteps, ndim))
            mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                              zip(*np.percentile(np.exp(flat), [16, 50, 84],
                                  axis=0)))
            gpdata = np.exp(mcmc_result[-1])
            GP_periods.append(mcmc_result[-1][0])
            gerrm.append(mcmc_result[-1][1])
            gerrp.append(mcmc_result[-1][2])
            gp_samps[i, :] = np.exp(np.random.choice(flat[:, -1], 50))
        except:
            GP_periods.append(0)
            gerrm.append(0)
            gerrp.append(0)
            gp_samps[i, :] = np.zeros(50)

        # load pgram results
        ps, pgram = np.genfromtxt("{0}/{1}_pgram.txt".format(fn,
                                                    str(int(id)).zfill(4))).T
        peaks = np.array([j for j in range(1, len(ps)-1) if pgram[j-1] <
                         pgram[j] and pgram[j+1] < pgram[j]])
        period = ps[pgram==max(pgram[peaks])][0]
        p_periods.append(period)

        # load myacf results
        myacf = np.genfromtxt("{0}/{1}_myacf_result.txt".format(fn,
                                            str(int(id)).zfill(4)))[0]
        my_p.append(myacf)

    # format and save results
    acf_results = np.vstack((np.array(acf_periods), np.array(aerrs)))
    gp_results = np.vstack((np.array(GP_periods), np.array(gerrm),
                           np.array(gerrp)))
    np.savetxt("{0}/acf_results.txt".format(fn), acf_results.T)
    np.savetxt("{0}/gp_results.txt".format(fn), gp_results.T)
    np.savetxt("{0}/pgram_results.txt".format(fn), p_periods)
    np.savetxt("{0}/myacf_results.txt".format(fn), np.array(my_p).T)
    return acf_results, gp_results, np.array(p_periods), np.array(my_p), \
            gp_samps

def compare_truth(N, coll=True):
    """
    make a plot comparing the truths to the measurements
    """
    ids, true_p, true_a = np.genfromtxt("{0}/true_periods.txt".format(fn)).T

    if coll:
        acf_results, gp_results, p_p, my_p, gp_samps = collate(N)
        acf_p, aerr = acf_results
        gp_p, gerrm, gerrp = gp_results
    else:
        acf_p, aerr = np.genfromtxt("{0}/acf_results.txt".format(fn)).T
        gp_p, gerrm, gerrp = np.genfromtxt("{0}/gp_results.txt".format(fn)).T
        _, p_p = np.genfromtxt("{0}/periodogram_results.txt".format(fn)).T
        my_p = np.genfromtxt("{0}/myacf_results.txt".format(fn)).T

    print len(true_p[:N]), "truths found"
    print len(acf_p[:N]), "acfs found"
    print len(gp_p[:N]), "gps found"
    print len(p_p[:N]), "periodograms found"
    print len(my_p[:N]), "myacfs found"

    m = acf_p[:N] > 0
    true_p[:N] = true_p[:N][m]
    acf_p[:N] = acf_p[:N][m]
    gp_p[:N] = gp_p[:N][m]
    p_p[:N] = p_p[:N][m]
    my_p[:N] = my_p[:N][m]

    plt.clf()
    xs = np.linspace(0, 100, 100)
    plt.plot(xs, xs, ".5", ls="--")
    plt.scatter(true_p[:N], acf_p[:N], color=cols.pink, s=8, alpha=.7)
    plt.errorbar(true_p[:N], gp_p[:N], yerr=(gerrp[:N], gerrm[:N]),
                 color=cols.blue, #alpha=.7,
                 fmt="o", label="$\mathrm{GP}$", capsize=0, ms=4,
                 mec=cols.blue)
    plt.scatter(true_p[:N], p_p[:N], color=cols.maroon, s=20, marker="^",
                label="$\mathrm{Periodogram}$")

    plt.subplots_adjust(bottom=.2)
    plt.xlim(0, 60)
    plt.ylim(0, 80)
    plt.legend(loc="upper left")
    plt.xlabel("$\mathrm{True~period~(days)}$")
    plt.ylabel("$\mathrm{Measured~period~(days)}$")
    plt.savefig("compare.pdf")

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
    fits_path = "{0}/{1}/*llc.fits".format(p, id)
    fnames = np.sort(glob.glob(fits_path))
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

    # noise-free simulations
    N = 2
    ids = range(N)
    pool = Pool()
    pool.map(acf_pgram_GP_sim, ids)

#     # noisy simulations
#     N = 2
#     ids = range(N)
#     pool = Pool()
#     pool.map(acf_pgram_GP_noisy, ids)

#     # real lcs
#     data = np.genfromtxt("data/garcia.txt", skip_header=1).T
#     kids = [str(int(i)).zfill(9) for i in data[0]]
#     pool = Pool()
#     pool.map(acf_pgram_GP, kids)
