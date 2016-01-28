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

def simulate(id, time, period, gen_type="s", plot=False):
    """
    Simulate noise free light curves with the same time-sampling as the real
    target that you are going to inject into.
    id: the kepler id that the time values are taken from.
    time: an array of time values.
    takes an array of periods and amplitudes, periods
    periods in days.
    gen_type == "s" for stellar and "GP" for gaussian process
    """

    if gen_type == "s":
        res0, res1 = mklc.mklc(time, p=period)
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

    y = simflux - np.mean(simflux)
    return y / np.std(y)

def run(pmin, pmax, amin, amax, nsim, plot=False):
    ids = np.genfromtxt("data/quiet_kepler_ids.txt", dtype=int).T

    r_ps, r_as = [], []
    n = 0
    for number, j in enumerate(ids):  # loop over the real kepler light curves
        id = str(j)
        print(id)
        d = "/home/angusr/.kplr/data/lightcurves"
        fnames = np.sort(glob.glob("{0}/{1}/*llc.fits".format(d, id.zfill(9))))
        x, y, yerr = load_kepler_data(fnames)  # load the time arrays
        x -= x[0]
        std = np.std(y)
        y, yerr = y / std, yerr / std

        # generate array of periods and amps
        periods = np.exp(np.random.uniform(np.log(pmin), np.log(pmax), nsim))
        amps = np.exp(np.random.uniform(np.log(amin), np.log(amax), nsim))
        r_ps.append(periods)
        r_as.append(amps)

        for i, p in enumerate(periods):  # loop over periods
            print i, "of", len(periods), "periods", number, "of" len(ids), \
                    "stars"
            new_y = simulate(id, x, p, plot=True)  # sim
            new_std = np.std(new_y)
            new_y /= new_std
            noisy_y = y + new_y * amps[i]

            noisy_data = np.vstack((x, noisy_y, yerr))
            data = np.vstack((x, new_y * amps[i], yerr))
            fn = "simulations/kepler_injections"
            fn2 = "simulations/noise-free"
            np.savetxt("{0}/{1}.txt".format(fn, str(n).zfill(4)), noisy_data.T)
            np.savetxt("{0}/{1}.txt".format(fn2, str(n).zfill(4)), data.T)

            if plot:
                plt.clf()
                plt.plot(x, noisy_y, "k.")
                plt.plot(x, new_y * amps[i], "r.")
                plt.title("p = {0}, a = {1}".format(p, amps[i]))
                plt.savefig("{0}/{1}".format(fn, str(n).zfill(4)))
                print("{0}/{1}.png".format(fn, str(n).zfill(4)))

            n += 1

    # record the truths
    r_ps = np.array([i for j in r_ps for i in j])
    r_as = np.array([i for j in r_as for i in j])
    data = np.vstack((np.arange(len(r_ps)), r_ps, r_as))
    np.savetxt("simulations/kepler_injections/true_periods_amps.txt", data.T)

if __name__ == "__main__":
    pmin, pmax =.5, 100.
    amin, amax = 1e-1, 10
    nsim = 1
    run(pmin, pmax, amin, amax, nsim, plot=True)
