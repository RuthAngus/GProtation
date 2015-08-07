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

    def easy_fit(self):
        """
        This function spits out a posterior for Kepler data
        """
        kepler_cadence = .02043365
        p_init, acf, lags = simple_acf(x, y, kepler_cadence)

    def MCMC(self, theta_init, plims, burnin, run, nwalkers):
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
        plims (tuple): the upper and lower bounds for a flat (in log-space)
        prior.
        burnin (integer) is the number of steps during burn-in.
        run (integer) is the number of steps for the full run.
        nwalkers is the number of walkers.
        This function returns the emcee sampler object.
        See dfm.io/emcee for emcee documentation.
        """

        # run MCMC
        p_init = theta_init[-1]
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
        Here is a prior function if you want to use it!
        theta is the array of parameters.
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

    def neglnlike(self, theta):
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

    # Dan Foreman-Mackey's acf function
    def dan_acf(self, axis=0, fast=False):
        """
        Estimate the autocorrelation function of a time series using the FFT.
        :param x:
            The time series. If multidimensional, set the time axis using the
            ``axis`` keyword argument and the function will be computed for every
            other axis.
        :param axis: (optional)
            The time axis of ``x``. Assumed to be the first axis if not specified.
        :param fast: (optional)
            If ``True``, only use the largest ``2^n`` entries for efficiency.
            (default: False)
        """
        self.x = np.atleast_1d(self.x)
        m = [slice(None), ] * len(self.x.shape)

        # For computational efficiency, crop the chain to the largest power of
        # two if requested.
        if fast:
            n = int(2**np.floor(np.log2(self.x.shape[axis])))
            m[axis] = slice(0, n)
            x = self.x
        else:
            n = self.x.shape[axis]

        # Compute the FFT and then (from that) the auto-correlation function.
        f = np.fft.fft(self.x-np.mean(self.x, axis=axis), n=2*n, axis=axis)
        m[axis] = slice(0, n)
        acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[m].real
        m[axis] = 0
        return acf / acf[m]

    def simple_acf(self, cadence):
        """
        A simple implementation of the ACF method.
        Takes x and y, returns period, smoothed acf, lags and flags
        """
        # fit and subtract straight line
        AT = np.vstack((self.x, np.ones_like(self.x)))
        ATA = np.dot(AT, AT.T)
        m, b = np.linalg.solve(ATA, np.dot(AT, self.y))
        ny -= m*self.x + b

        # perform acf
        acf = dan_acf(ny)

        # smooth with Gaussian kernel convolution
        Gaussian = lambda x,sig: 1./np.sqrt(2*np.pi*sig**2) * \
                     np.exp(-0.5*(x**2)/(sig**2)) #define a Gaussian
        conv_func = Gaussian(np.arange(-28,28,1.), 9.)
        acf_smooth = np.convolve(acf, conv_func, mode='same')

        lags = np.arange(len(acf)) * cadence

        # find all the peaks
        peaks = np.array([i for i in range(1, len(lags)-1)
                         if acf_smooth[i-1] < acf_smooth[i] and
                         acf_smooth[i+1] < acf_smooth[i]])

        # throw the first peak away
        peaks = peaks[1:]

        # find the first and second peaks
        if acf_smooth[peaks[0]] > acf_smooth[peaks[1]]:
            period = lags[peaks[0]]
            h = acf_smooth[peaks[0]]  # peak height
        else:
            period = lags[peaks[1]]
            h = acf_smooth[peaks[1]]

        return period, acf_smooth, lags
