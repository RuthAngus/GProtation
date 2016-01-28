import numpy as np
import matplotlib.pyplot as plt
import h5py
from plotstuff import params, colours
reb = params()
cols = colours()

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

if __name__ == "__main__":
    compare_truth(10)
