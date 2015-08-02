from __future__ import print_function
import numpy as np
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import emcee
import triangle
import h5py
from Kepler_ACF import corr_run
from tools import bin_data

class gisela(object):

    def __init__(self, x, y, yerr):
        self.x = x
        self.y = y
        self.yerr = yerr

    def MCMC(self, id, theta_init, lower=.7, upper=1.5,
            burnin=1000, run=1500):
        """
        Use this function if you want to remain agnostic about the
        values of the less important hyperparameters and you want
        to initialise with a period measured using the ACF method.
        It usually seems ok to set the initial hyperparameter values to be
        1.
        id (str, int): the name of the star or whatever you want to save
        the output as.
        lower (float): the multiple of p_init you want to use for the lower
        bound in the prior function.
        upper (float): the multiple of p_init you want to use for the upper
        bound in the prior function.
        I set the upper and lower bounds on the prior to be .7 and
        1.5 times the initial period.
        """
        """
        This function runs emcee in order to sample the posterior of your
        hyperparameters or find the maxima, etc.
        theta_init (array): logarithmic initial parameters
        [A, gamma, l, sigma, period]
        A = amplitude.
        gamma = scaling factor in the exponential-sine kernel.
        l = length scale of the squared-exponential kernel
        sigma = white noise multiplier
        period = period (obv). In the same units as your 'xs'.
        It takes an initial guess for your parameters (theta_init).
        These must be logarithmic.
        It takes your data: x, y, yerr.
        plims are your (log) upper and lower prior bounds on the period.
        same units as your x values.
        burnin (integer) is the number of steps during burn-in.
        run (integer) is the number of steps for the full run.
        In practise I find that 32 walkers are fine, so I've fixed this
        number.
        This function returns the emcee sampler object.
        See dfm.io/emcee for emcee documentation.
        """

        # run MCMC
        p_init = theta_init[-1]
        plims = np.log([p_init*lower, p_init*upper])

        ndim, nwalkers = len(theta_init), 32
        p0 = [theta_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
        args = [plims]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=args)

        print("burning in...")
        p0, lp, state = sampler.run_mcmc(p0, burnin)
        sampler.reset()
        print("production run...")
        p0, lp, state = sampler.run_mcmc(p0, run)
        return sampler

    def lnprior(self, theta, plims):
        """
        theta is an array of parameters
        plims is a tuple containing the (log) lower and upper limits for
        the rotation period. These are logarithmic!
        These form the upper and lower bounds of a prior that is
        flat in log-space.
        The other parameters should be well behaved, so I use very
        broad priors for them.
        """
        if -20 < theta[0] < 20 and -20 < theta[1] < 20 \
                and -20 < theta[2] < 20 and -20 < theta[3] < 20 \
                and plims[0] < theta[4] < plims[1]:
            return 0.
        return -np.inf

    def lnprob(self, theta, plims):
        """
        The posterior function
        """
        return self.lnlike(theta) + self.lnprior(theta, plims)

    def lnlike(self, theta):
        """
        The log-likelihood function.
        Uses a squared exponential times an exponential-sine
        kernel function.
        theta is the array of parameters:
        theta = [A, gamma, l, sigma, period]
        A = amplitude.
        gamma = scaling factor in the exponential-sine kernel.
        l = length scale of the squared-exponential kernel
        sigma = white noise multiplier
        period = period (obv). In the same units as your 'xs'.

        These must be logarithmic.
        """

        theta = np.exp(theta)
        k = theta[0] * ExpSquaredKernel(theta[1]) \
                * ExpSine2Kernel(theta[2], theta[4])
        gp = george.GP(k)
        try:
            gp.compute(self.x, np.sqrt(theta[3]+self.yerr**2))
        except (ValueError, np.linalg.LinAlgError):
            return 10e25
        return gp.lnlikelihood(self.y, quiet=True)

    def neglnlike(theta):
        """
        The negative version of the log-likelihood function.
        For convenience, in case you ever want to optimise instead
        of sampling.
        You could also do gp.optimize()
        """
        theta = np.exp(theta)
        k = theta[0] * ExpSquaredKernel(theta[1]) \
                * ExpSine2Kernel(theta[2], theta[4])
        gp = george.GP(k)
        try:
            gp.compute(self.x, np.sqrt(theta[3]+self.yerr**2))
        except (ValueError, np.linalg.LinAlgError):
            return 10e25
        return -gp.lnlikelihood(self.y, quiet=True)
