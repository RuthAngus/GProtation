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
                burnin=5000, run=10000, npts=npts, cutoff=cutoff,
                sine_kernel=True, acf=False, runMCMC=runMCMC, plot=plot)

def collate(N, save=False):
    """
    assemble all the period measurements here
    """
    ids, true_p, true_a = np.genfromtxt("simulations/true_periods.txt").T
    acf_periods, aerrs, GP_periods, gerrp, gerrm = [], [], [], [], []
    p_periods, pids = [], []  # periodogram results. pids is just a place hold
    my_p = []
    gp_samps = np.zeros((len(ids), 50))
    for i, id in enumerate(ids[:N]):

        # load ACF results
        try:
            acfdata = np.genfromtxt("simulations/%s_result.txt"
                                    % int(id)).T
            acf_periods.append(acfdata[0])
            aerrs.append(acfdata[1])
        except:
            acf_periods.append(0)
            aerrs.append(0)

        try:
            # load GP results
            with h5py.File("sine/%s_samples.h5" % str(int(id)).zfill(4), "r") as f:
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
        ps, pgram = np.genfromtxt("simulations/%s_pgram.txt"
                                  % str(int(id)).zfill(4)).T
        peaks = np.array([j for j in range(1, len(ps)-1) if pgram[j-1] <
                         pgram[j] and pgram[j+1] < pgram[j]])
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
    if save:
        np.savetxt("simulations/acf_results.txt", acf_results.T)
        np.savetxt("simulations/gp_results.txt", gp_results.T)
        np.savetxt("simulations/pgram_results.txt", p_periods)
        np.savetxt("simulations/myacf_results.txt", np.array(my_p).T)
    return acf_results, gp_results, np.array(p_periods), np.array(my_p), gp_samps

def compare_truth(N, coll=True, save=False):
    """
    make a plot comparing the truths to the measurements
    """
    ids, true_p, true_a = np.genfromtxt("simulations/true_periods.txt").T

    if coll:
        acf_results, gp_results, p_p, my_p, gp_samps = collate(N, save=save)
        acf_p, aerr = acf_results
        gp_p, gerrm, gerrp = gp_results
    else:
        acf_p, aerr = np.genfromtxt("simulations/acf_results.txt").T
        gp_p, gerrm, gerrp = np.genfromtxt("simulations/gp_results.txt").T
        _, p_p = np.genfromtxt("simulations/periodogram_results.txt").T
        my_p = np.genfromtxt("simulations/myacf_results.txt").T

    s = np.log(true_a[:N]*1e15)
    s = true_a[:N]*1000

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
# #     plt.errorbar(true_p[:N], acf_p[:N], yerr=aerr[:N], color=cols.pink, fmt=".",
# #                  label="$\mathrm{ACF}$", capsize=0, alpha=.7)

# #     plt.scatter(true_p[:N], acf_p[:N], color=cols.pink, s=8, alpha=.7,
#     plt.scatter(true_p[:N], acf_p[:N], color=cols.orange, s=13,
#                 label="$\mathrm{ACF}$", marker="s")

#     m = gerrp[:N] < 5
#     plt.errorbar(true_p[:N][m], gp_p[:N][m], yerr=(gerrp[:N][m], gerrm[:N][m]),
#                  color=cols.blue, #alpha=.7,
#                  fmt="o", label="$\mathrm{GP}$", capsize=0, ms=4,
#                  mec=cols.blue)
# #     plt.scatter(true_p[:N], gp_p[:N], color=cols.blue, s=20, alpha=.7,
# #                 label="$\mathrm{GP}$")

# #     true_samps = np.zeros((N, 50))
# #     for i in range(50):
# #         true_samps[:, i] = true_p[:N]
# #     plt.scatter(true_samps[:N, :], gp_samps[:N, :], color=cols.blue, s=20, alpha=.2,
# #                 label="$\mathrm{GP}$")

    plt.scatter(true_p[:N], p_p[:N], color=cols.maroon, s=20, marker="^",
                label="$\mathrm{Periodogram}$")
# #     plt.scatter(true_p[:N], my_p[:N], color=cols.green, s=20, alpha=.7,
# #                 label="$\mathrm{simple}$")

# #     plt.plot(xs, 2*xs, ".5", ls="--")
# #     plt.plot(xs, .5*xs, ".5", ls="--")
    plt.subplots_adjust(bottom=.2)
    plt.xlim(0, 60)
# #     plt.xlim(20, 60)
    plt.ylim(0, 80)
# #     plt.ylim(20, 80)
    plt.legend(loc="upper left")
    plt.xlabel("$\mathrm{True~period~(days)}$")
    plt.ylabel("$\mathrm{Measured~period~(days)}$")
#     plt.savefig("compare2")
#     plt.savefig("compare_acf")
#     plt.savefig("compare_gp")
    plt.savefig("compare_pgram")

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
    compare_truth(300, coll=True, save=False)
