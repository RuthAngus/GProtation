import numpy as np
import matplotlib.pyplot as plt
import mklc
import pyfits
import fitsio
import scipy.interpolate as spi

def simulate(id, pmin=.5, pmax=100., amin=1e-3, amax=1e-1, nsim=100):
    """
    pmin and pmax in days, amin and amax in ppm.
    """
    periods = np.exp(np.random.uniform(np.log(pmin), np.log(pmax), nsim))
    amps = np.exp(np.random.uniform(np.log(amin), np.log(amax), nsim))
    np.savetxt("true_periods.txt", np.transpose((np.arange(nsim), periods,
               amps)))

    # load test star data FIXME: I'm just using the time values
    # because this is not a quiet star
    x, y, yerr = np.genfromtxt("%s_lc.txt" % id).T
    time = x
    std = 1e-5
    yerr = np.ones_like(time)*std
    flux = np.zeros_like(x) + np.random.randn(len(time))*std

    for i, p in enumerate(periods):
        print i, "of ", len(periods), "\n"
        print "amps = ", amps[i]
        res0, res1 = mklc.mklc(time, p=p)
        nspot, ff, amp_err = res0
        time, area_tot, dF_tot, dF_tot0 = res1
        simflux = dF_tot0 / np.median(dF_tot0) - 1

        np.savetxt("simulations/%s.txt" % i, np.transpose((time, simflux)))

        plt.clf()
        plt.plot(time, simflux*amps[i]+flux, "k.")
        plt.savefig("simulations/%s" % i)
        plt.title("p = %s, a = %s" % (p, amps[i]))

if __name__ == "__main__":
    simulate("../kepler452b/8311864", pmin=.5, pmax=100.)
