from __future__ import print_function
import numpy as np
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import emcee
import triangle
import h5py
from Kepler_ACF import corr_run
from tools import bin_data

class GProtation(object):

    def __init__(self, x, y, yerr):
        self.x = x
        self.y = y
        self.yerr = yerr

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
