from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import time
import os
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
from gatspy.periodic import LombScargle
import emcee
try:
    import corner
except:
    import triangle
import h5py
from Kepler_ACF import corr_run

plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 20,
           'xtick.labelsize': 15,
           'ytick.labelsize': 15,
           'text.usetex': True}
plt.rcParams.update(plotpar)


class gprotation(object):

    def __init__(self, x, y, yerr, id, RESULTS_DIR):
            self.x = x
            self.y = y
            self.yerr = yerr
            self.id = id
            self.RESULTS_DIR = RESULTS_DIR

    def calc_p_init(self, which_period="acf"):

        print("Calculating ACF")
        acf_period, err, lags, acf = corr_run(self.x, self.y, self.yerr,
                                              self.id, self.RESULTS_DIR)
        print("acf period, err = ", acf_period, err)

        print("Calculating periodogram")
        ps = np.arange(.1, 100, .1)
        model = LombScargle().fit(self.x, self.y, self.yerr)
        pgram = model.periodogram(ps)
        plt.clf()
        plt.plot(ps, pgram)
        plt.savefig(os.path.join(self.RESULTS_DIR,
                                 "{0}_pgram".format(self.id)))
        print("saving figure ",
              os.path.join(self.RESULTS_DIR, "{0}_pgram".format(self.id)))

        peaks = np.array([i for i in range(1, len(ps)-1) if pgram[i-1] <
                          pgram[i] and pgram[i+1] < pgram[i]])
        pgram_period = ps[pgram == max(pgram[peaks])][0]
        print("pgram period = ", pgram_period, "days")

        if which_period == "acf":
            p_init, perr = acf_period, err
        elif which_period == "pgram":
            p_init, perr = pgram_period, pgram_period * .1
        else:
            print("which_period must equal 'acf' or 'pgram'")

        return p_init, perr

    def fit(self, p_init, burnin=500, nwalkers=12, nruns=10):

        theta_init = np.log([np.exp(-5), np.exp(7), np.exp(.6), np.exp(-16),
                             p_init])

        # Set up MCMC parameters
        plims = np.log([.1*p_init, 5*p_init])
        print("total number of points = ", len(self.x))
        runs = np.zeros(nruns) + 500
        ndim = len(theta_init)
        p0 = [theta_init + 1e-4 * np.random.rand(ndim) for i in
              range(nwalkers)]
        args = (plims)

        # time the lhf call
        start = time.time()
        print("lnprob = ", self.lnprob(theta_init, plims))
        end = time.time()
        tm = end - start
        print("1 lhf call takes ", tm, "seconds")
        print("burn in will take", tm * nwalkers * burnin, "s")
        print("each run will take", tm * nwalkers * runs[0]/60, "mins")
        print("total = ", (tm * nwalkers * np.sum(runs) + tm * nwalkers *
                           burnin)/60, "mins")

        # Run the MCMC
        lp = self.lnprob(theta_init, plims)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp,
                                        args=args)
        print("burning in...")
        p0, _, state = sampler.run_mcmc(p0, burnin)

        sample_array = np.zeros((nwalkers, sum(runs), ndim))
        for i, run in enumerate(runs):
            sampler.reset()
            print("production run, {0} steps".format(int(run)))
            start = time.time()
            p0, _, state = sampler.run_mcmc(p0, run)
            end = time.time()
            print("time taken = ", (end - start)/60, "minutes")

            # save samples
            sample_array[:, sum(runs[:i]):sum(runs[:(i+1)]), :] \
                = np.array(sampler.chain)
            f = h5py.File(os.path.join(self.RESULTS_DIR,
                                       "{0}.h5".format(self.id)), "w")
            data = f.create_dataset("samples",
                                    np.shape(sample_array[:,
                                                          :sum(runs[:(i+1)]),
                                                          :]))
            data[:, :] = sample_array[:, :sum(runs[:(i+1)]), :]
            f.close()

            # make various plots
            with h5py.File(os.path.join(self.RESULTS_DIR,
                                        "{0}.h5".format(self.id)), "r") as f:
                samples = f["samples"][...]
            mcmc_result = self.make_plot(samples, traces=True, tri=True,
                                         prediction=True)
        return mcmc_result

    def lnprior(self, theta, plims):
        """
        plims is a tuple, list or array containing the lower and upper limits
        for the rotation period. These are logarithmic!
        theta = A, l, Gamma, s, P
        """
        if -20 < theta[0] < 20 and theta[4] < theta[1] and \
                -20 < theta[2] < 20 and -20 < theta[3] < 20 \
                and plims[0] < theta[4] < plims[1]:
            return 0.
        return -np.inf

    def lnprob(self, theta, plims):
        return self.lnlike(theta) + self.lnprior(theta, plims)

    def lnlike(self, theta):
        theta = np.exp(theta)
        k = theta[0] * ExpSquaredKernel(theta[1]) \
            * ExpSine2Kernel(theta[2], theta[4])
        gp = george.GP(k, solver=george.HODLRSolver)
        try:
            gp.compute(self.x, np.sqrt(theta[3]**2 + self.yerr**2))
        except (ValueError, np.linalg.LinAlgError):
            return 10e25
        return gp.lnlikelihood(self.y, quiet=True)

    def neglnlike(self, theta):
        theta = np.exp(theta)
        k = theta[0] * ExpSquaredKernel(theta[1]) \
            * ExpSine2Kernel(theta[2], theta[4])
        gp = george.GP(k)
        try:
            gp.compute(self.x, np.sqrt(theta[3] + self.yerr**2))
        except (ValueError, np.linalg.LinAlgError):
            return 10e25
        return -gp.lnlikelihood(self.y, quiet=True)

    def make_plot(self, sampler, DIR, traces=False, tri=False,
                  prediction=True):

        plt.clf()
        plt.plot(self.x, self.y, "k.")
        plt.savefig("test")

        nwalkers, nsteps, ndims = np.shape(sampler)
        flat = np.reshape(sampler, (nwalkers * nsteps, ndims))
        mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                          zip(*np.percentile(flat, [16, 50, 84], axis=0)))
        mcmc_result = np.array([i[0] for i in mcmc_result])
        print("\n", np.exp(np.array(mcmc_result[-1])), "period (days)", "\n")
        print(mcmc_result)
        np.savetxt(os.path.join(DIR, "{0}_result.txt".format(self.id)),
                   mcmc_result)

        fig_labels = ["ln(A)", "ln(l)", "ln(G)", "ln(s)", "ln(P)"]

        if traces:
            print("Plotting traces")
            for i in range(ndims):
                plt.clf()
                plt.plot(sampler[:, :, i].T, 'k-', alpha=0.3)
                plt.ylabel(fig_labels[i])
                plt.savefig(os.path.join(DIR, "{0}_{1}.png".format(self.id,
                            fig_labels[i])))

        if tri:
            print("Making triangle plot")
            try:
                fig = corner.corner(flat, labels=fig_labels)
            except:
                fig = triangle.corner(flat, labels=fig_labels)
            fig.savefig(os.path.join(DIR, "{0}_triangle".format(self.id)))
            print(os.path.join(DIR, "{0}_triangle.png".format(self.id)))

        if prediction:
            print("plotting prediction")
            theta = np.exp(np.array(mcmc_result))
            k = theta[0] * ExpSquaredKernel(theta[1]) \
                * ExpSine2Kernel(theta[2], theta[4])
            gp = george.GP(k, solver=george.HODLRSolver)
            gp.compute(self.x-self.x[0], (self.yerr**2 + theta[3]**2)**.5)
            xs = np.linspace((self.x - self.x[0])[0],
                             (self.x - self.x[0])[-1], 1000)
            mu, cov = gp.predict(self.y, xs)
            plt.clf()
            plt.errorbar(self.x - self.x[0], self.y, yerr=self.yerr, fmt="k.",
                         capsize=0)
            plt.xlabel("$\mathrm{Time~(days)}$")
            plt.ylabel("$\mathrm{Normalised~Flux}$")
            plt.plot(xs, mu, color=cols.lightblue)
            plt.xlim(min(self.x - self.x[0]), max(self.x - self.x[0]))
            plt.savefig(os.path.join(DIR, "{0}_prediction".format(self.id)))
            print(os.path.join(DIR, "{0}_prediction.png".format(self.id)))



if __name__ == "__main__":

    DATA_DIR = "data/"
    RESULTS_DIR = "results/"

    import pyfits
    epic_id = 211000411
    fname = "hlsp_everest_k2_llc_{0}-c04_kepler_v1.0_lc.fits".format(epic_id)
    hdulist = pyfits.open(fname)
    t, flux = hdulist[1].data["TIME"], hdulist[1].data["FLUX"]
    out = hdulist[1].data["OUTLIER"]
    m = np.isfinite(t) * np.isfinite(flux) * (out < 1)
    med = np.median(flux[m])
    x, y = t[m], flux[m]/med - 1
    yerr = np.ones_like(y) * 1e-5

    xb, yb, yerrb = x[::10], y[::10], yerr[::10]

    GProt = gprotation(xb, yb, yerrb, epic_id, "results")
    p_init, perr = GProt.calc_p_init(which_period="pgram")

    mcmc_result = GProt.fit(p_init)
