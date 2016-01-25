import numpy as np
import matplotlib.pyplot as plt
import mklc
import pyfits
import fitsio
import scipy.interpolate as spi
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
from kepler_data import load_kepler_data
import glob

def simulate(id, time, periods, gen_type="s", plot=False):
    """
    Simulate noise free light curves with the same time-sampling as the real
    target that you are going to inject into.
    id: the kepler id that the time values are taken from.
    time: an array of time values.
    takes an array of periods and amplitudes, periods
    periods in days.
    gen_type == "s" for stellar and "GP" for gaussian process
    """

    indices, ids, ps  = [], [], []  # lists for saving truths
    for i, p in enumerate(periods):
        print i, "of ", len(periods), "\n"
        if gen_type == "s":
            res0, res1 = mklc.mklc(time, p=p)
            nspot, ff, amp_err = res0
            time, area_tot, dF_tot, dF_tot0 = res1
            simflux = dF_tot0 / np.median(dF_tot0) - 1

        elif gen_type == "GP":  # if performing sims with GPs
            thetas = np.zeros((nsim, 5))
            thetas[:, 0] = np.exp(np.random.uniform(-6, -4))
            thetas[:, 1] = np.exp(np.random.uniform(4, 7))
            thetas[:, 2] = np.exp(np.random.uniform(-1.5, 1.5))
            thetas[:, 3] = np.exp(np.random.uniform(-11, -8))
            thetas[:, 4] = periods
            k = thetas[i, 0] * ExpSquaredKernel(thetas[i, 1]) \
                    * ExpSine2Kernel(thetas[i, 2], thetas[i, 4]) \
                    + WhiteKernel(std)
            gp = george.gp()
            gp.compute(time, yerr)
            simflux = gp.sample()

        np.savetxt("simulations/kepler_injections/%s_%s.txt"
                   % (str(int(i)).zfill(4), str(int(id)).zfill(9)),
                   np.transpose((time, simflux)))

        if plot:
            plt.clf()
            plt.plot(time, simflux, "k.")
            plt.title("p = %s" % p)
            plt.savefig("simulations/kepler_injections/%s_%s"
                        % (str(int(i)).zfill(4), str(int(id)).zfill(9)))
            print("simulations/kepler_injections/%s%s" % (i, gen_type))

        indices.append(i)
        ids.append(id)
        ps.append(p)

    return indices, ids, ps

if __name__ == "__main__":
    pmin, pmax =.5, 100.
#     amin, amax = 1e-7, 1e-5
    nsim = 2

    # generate arrays of periods and amplitudes
    periods = np.exp(np.random.uniform(np.log(pmin), np.log(pmax), nsim))

    r_inds, r_ids, r_ps = [], [], []
    ids = np.genfromtxt("data/quiet_kepler_ids.txt", dtype=int).T
    for i in ids:  # loop over the real kepler light curves
        id = str(i)
        print(id)
        d = "/home/angusr/.kplr/data/lightcurves"
        fnames = np.sort(glob.glob("{0}/{1}/*llc.fits".format(d, id.zfill(9))))
        x, _, _ = load_kepler_data(fnames)  # load the time arrays

        inds, ids, ps = simulate(id, x, periods, plot=True)  # sim
        r_inds.append(inds)  # record the truths
        r_ids.append(ids)
        r_ps.append(ps)

    # record the truths
    r_inds = np.array([i for j in r_inds for i in j])
    r_ids = np.array([int(i) for j in r_ids for i in j])
    r_ps = np.array([i for j in r_ps for i in j])
    data = np.vstack((r_inds, r_ids, r_ps))
    np.savetxt("simulations/kepler_injections/true_periods.txt", data.T)
