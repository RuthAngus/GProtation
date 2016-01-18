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

def simulate(id, time, periods, amps, gen_type="s", plot=False):
    """
    Simulate noise free light curves with the same time-sampling as the real
    target that you are going to inject into.
    id: the kepler id that the time values are taken from.
    time: an array of time values.
    takes an array of periods and amplitudes, periods, amps
    periods in days, amplitudes in ppm.
    gen_type == "s" for stellar and "GP" for gaussian process
    """

    for i, p in enumerate(periods):
        print i, "of ", len(periods), "\n"
        print "amps = ", amps[i]
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
                   % (str(int(i)).zfill(4), gen_type),
                   np.transpose((time, simflux)))

        if plot:
            plt.clf()
            plt.plot(time, simflux*amps[i], "k.")
            plt.savefig("simulations/kepler_injections/%s%s" % (i, gen_type))
            print("simulations/kepler_injections/%s%s" % (i, gen_type))
            plt.title("p = %s, a = %s" % (p, amps[i]))

if __name__ == "__main__":
    pmin, pmax =.5, 100.
    amin, amax = 1e-3, 1e-1
    nsim = 100

    periods = np.exp(np.random.uniform(np.log(pmin), np.log(pmax), nsim))
    amps = np.exp(np.random.uniform(np.log(amin), np.log(amax), nsim))
    np.savetxt("simulations/kepler_injections/true_periods.txt",
               np.transpose((np.arange(nsim), periods, amps)))

    ids = np.genfromtxt("quiet_kepler_ids.txt", dtype=int).T
    for i in ids:
        id = str(i)
        print(id)
        d = "/home/angusr/.kplr/data/lightcurves"
        fnames = np.sort(glob.glob("{0}/{1}/*llc.fits".format(d, id.zfill(9))))
        x, _, _ = load_kepler_data(fnames)

        simulate(id, x, periods, amps, plot=True)
