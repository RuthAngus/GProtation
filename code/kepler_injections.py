import numpy as np
import matplotlib.pyplot as plt
import mklc
import pyfits
import fitsio
import scipy.interpolate as spi
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel

def simulate(id, x, pmin=.5, pmax=100., amin=1e-3, amax=1e-1, nsim=100,
             gen_type="s", kepler=False, plot=False):
    """
    Simulate noise free light curves with the same time-sampling as the real
    target that you are going to inject into.
    pmin and pmax in days, amin and amax in ppm.
    """
    periods = np.exp(np.random.uniform(np.log(pmin), np.log(pmax), nsim))
    amps = np.exp(np.random.uniform(np.log(amin), np.log(amax), nsim))
    np.savetxt("simulations/true_periods.txt", np.transpose((np.arange(nsim),
               periods, amps)))

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

        np.savetxt("simulations/%s_%s.txt" % (str(int(i)).zfill(4), gen_type),
                   np.transpose((time, simflux)))

        if plot:
            plt.clf()
            plt.plot(time, simflux*amps[i]+flux, "k.")
            plt.savefig("simulations/%s%s" % (i, gen_type))
            plt.title("p = %s, a = %s" % (p, amps[i]))
            assert 0

if __name__ == "__main__":
    import glob
    from kepler_data import load_kepler_data

    ids = np.genfromtxt("kepler_ids.txt", dtype=int).T
    for i in ids[1:]:
        id = str(i)
        print(id)
        d = "/home/angusr/.kplr/data/lightcurves"
        fnames = np.sort(glob.glob("{0}/{1}/*llc.fits".format(d, id.zfill(9)))
        x, y, yerr = load_kepler_data(fnames)
        plt.clf()
        m = (x > 150) * (x < 400)
#         m = x < 500
        plt.plot(x[m], y[m], "k.")
        plt.savefig("simulations/kepler_injections/{0}".format(id.zfill(9)))

#     simulate("../kepler452b/8311864", pmin=.5, pmax=100., amin=1e-5, amax=1e-2,
#              nsim=100, gen_type="GP", plot=True)
