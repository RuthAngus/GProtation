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
from simple_acf import simple_acf

plotpar = {'axes.labelsize': 22,
           'font.size': 22,
           'legend.fontsize': 22,
           'xtick.labelsize': 22,
           'ytick.labelsize': 22,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def my_acf(N):
    ids = np.arange(N)
    periods = []
    for id in ids:
        print "\n", id, "of", N
        x, y = np.genfromtxt("simulations/%s.txt" % str(int(id)).zfill(4)).T
        period, acf, lags, flag = simple_acf(x, y)
        periods.append(period)
        print period
        np.savetxt("simulations/%s_myacf_result.txt" % str(int(id)).zfill(4),
                   np.ones(5)*period)
    np.savetxt("simulations/myacf_results.txt",
               np.transpose((ids, np.array(periods))))

def periodograms(N, plot=False, savepgram=True):

    ids = np.arange(N)
    periods = []
    for id in ids:
        print "\n", id, "of", N

        # load simulated data
        x, y = np.genfromtxt("simulations/%s.txt" % str(int(id)).zfill(4)).T
        yerr = np.ones_like(y) * 1e-5

        # initialise with acf
        try:
            p_init = np.genfromtxt("simulations/%s_result.txt"
                                   % int((id)))
        except:
            corr_run(x, y, yerr, int(id), "simulations",
                     saveplot=False)
            p_init = np.genfromtxt("simulations/%s_result.txt" % int(id))
        print "acf period, err = ", p_init

        ps = np.linspace(p_init[0]*.1, p_init[0]*4, 1000)
        model = LombScargle().fit(x, y, yerr)
        pgram = model.periodogram(ps)

        # find peaks
        peaks = np.array([i for i in range(1, len(ps)-1) if pgram[i-1] <
                         pgram[i] and pgram[i+1] < pgram[i]])
        period = ps[pgram==max(pgram[peaks])][0]
        periods.append(period)
        print "pgram period = ", period

        if plot:
            plt.clf()
            plt.plot(ps, pgram)
            plt.axvline(period, color="r")
            plt.savefig("simulations/%s_pgram" % str(int(id)).zfill(4))

        if savepgram:
            np.savetxt("simulations/%s_pgram.txt" % str(int(id)).zfill(4),
                       np.transpose((ps, pgram)))

    np.savetxt("simulations/periodogram_results.txt",
               np.transpose((ids, periods)))
    return periods

def recover_injections(start, stop, N=1000, runMCMC=True, plot=False):
    """
    run MCMC on each star, initialising with the ACF period
    Next you could try running for longer, using more of the lightcurve,
    subsampling less, etc
    """
#     ids, true_ps, true_as = np.genfromtxt("simulations/true_periods.txt").T
    ids = range(N)

    if start == 0: id_list = ids[:stop]
    elif stop == N: id_list = ids[start:]
    else: id_list = ids[start:stop]
    for id in id_list:
        print "\n", "star id = ", id

        # load simulated data
        x, y = np.genfromtxt("simulations/%s.txt" % str(int(id)).zfill(4)).T
        yerr = np.ones_like(y) * 1e-5

#         print "true period = ", true_ps[int(id)]

        # initialise with acf
        try:
            p_init = np.genfromtxt("simulations/%s_result.txt" % id)
        except:
            corr_run(x, y, yerr, id, "simulations")
            p_init = np.genfromtxt("simulations/%s_result.txt" % id)
        print "acf period, err = ", p_init

        # run MCMC
        plims = [p_init[0]*.7, p_init[0]*1.5]
        npts = int(p_init[0] / 10. * 48)  # 10 points per period
        cutoff = 10 * p_init[0]
        fit(x, y, yerr, str(int(id)).zfill(4), p_init[0], np.log(plims),
                burnin=2000, run=5000, npts=npts, cutoff=cutoff,
                sine_kernel=True, acf=False, runMCMC=runMCMC, plot=plot)

def collate(N):
    """
    assemble all the period measurements here
    """
    acf_periods, aerrs, GP_periods, gerrp, gerrm = [], [], [], [], []
    p_periods, pids = [], []  # periodogram results. pids is just a place hold
    my_p = []
    for id in range(N):

        # load ACF results
        try:
            acfdata = np.genfromtxt("simulations/%s_result.txt"
                                    % int(id)).T
            acf_periods.append(acfdata[0])
            aerrs.append(acfdata[1])
        except:
            acf_periods.append(0)
            aerrs.append(0)

        # load GP results
        try:
            with h5py.File("sine/%s_samples.h5" % str(id).zfill(4), "r") as f:
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
        except:
            GP_periods.append(0)
            gerrm.append(0)
            gerrp.append(0)

        # load pgram results
        ps, pgram = np.genfromtxt("simulations/%s_pgram.txt"
                                  % str(int(id)).zfill(4)).T
        peaks = np.array([i for i in range(1, len(ps)-1) if pgram[i-1] <
                         pgram[i] and pgram[i+1] < pgram[i]])
        period = ps[pgram==max(pgram[peaks])][0]
        p_periods.append(period)

        # load myacf results
        myacf = np.genfromtxt("simulations/%s_myacf_result.txt"
                              % str(int(id)).zfill(4))[0]
        my_p.append(myacf)

    # format and save results
    acf_results = np.vstack((np.array(acf_periods), np.array(aerrs)))
    gp_results = np.vstack((np.array(GP_periods), np.array(gerrm),
                           np.array(gerrp)))
    np.savetxt("simulations/acf_results.txt", acf_results.T)
    np.savetxt("simulations/gp_results.txt", gp_results.T)
    np.savetxt("simulations/pgram_results.txt", p_periods)
    np.savetxt("simulations/myacf_results.txt", np.array(my_p).T)
    return acf_results, gp_results, p_periods, my_p

def compare_truth(N, coll=True):
    """
    make a plot comparing the truths to the measurements
    """
    ids, true_p, true_a = np.genfromtxt("simulations/true_periods.txt").T

    if coll:
        acf_results, gp_results, p_p, my_p = collate(N)
        acf_p, aerr = acf_results
        gp_p, gerrm, gerrp = gp_results
    else:
        acf_p, aerr = np.genfromtxt("simulations/acf_results.txt").T
        gp_p, gerrm, gerrp = np.genfromtxt("simulations/gp_results.txt").T
        _, p_p = np.genfromtxt("simulations/periodogram_results.txt").T
        _, my_p = np.genfromtxt("simulations/myacf_results.txt").T

    s = np.log(true_a[:N]*1e15)
    s = true_a[:N]*1000

    print len(true_p[:N]), "truths found"
    print len(acf_p), "acfs found"
    print len(gp_p), "gps found"
    print len(p_p), "periodograms found"
    print len(my_p), "periodograms found"

    plt.clf()
    xs = np.linspace(0, 100, 100)
    plt.errorbar(true_p[:N], acf_p, yerr=aerr, color=cols.pink, fmt=".",
                 label="$\mathrm{ACF}$", capsize=0, alpha=.7)
    plt.errorbar(true_p[:N], gp_p, yerr=(gerrp, gerrm), color=cols.blue,
                 fmt=".", label="$\mathrm{GP}$", capsize=0, alpha=.7)
#     plt.scatter(true_p[:N], p_p[:N], color=cols.orange, s=20, alpha=.7,
#                 label="$\mathrm{Pgram}$")
#     plt.scatter(true_p[:N], my_p[:N], color=cols.green, s=20, alpha=.7,
#                 label="$\mathrm{simple}$")
    plt.plot(xs, xs, ".7", ls="--")
    plt.plot(xs, 2*xs, ".7", ls="--")
    plt.plot(xs, .5*xs, ".7", ls="--")
    plt.xlim(0, 100)
    plt.ylim(0, 170)
    plt.legend(loc="best")
    plt.xlabel("$\mathrm{True~period~(days)}$")
    plt.ylabel("$\mathrm{Measured~period~(days)}$")
    plt.savefig("compare")

if __name__ == "__main__":

    # measure periods using the periodogram method
#     periodograms(N=500, plot=False)

    # measure periods using simple_acf
#     my_acf(500)

    # run full MCMC recovery
#     start = int(sys.argv[1])
#     stop = int(sys.argv[2])
#     recover_injections(start, stop, runMCMC=True, plot=False)

    # make comparison plot
    compare_truth(500, coll=True)
