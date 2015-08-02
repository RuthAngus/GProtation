from __future__ import print_function
import numpy as np
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import emcee
import triangle
import h5py
from Kepler_ACF import corr_run

class gisela(object)

    def __init__(self, x, y, yerr):
        self.x = x
        self.y = y
        self.yerr = yerr

    def lnprior(theta, plims):
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

    def lnprob(theta, x, y, yerr, plims):
        """
        The posterior function
        """
        return lnlike(theta, x, y, yerr) + lnprior(theta, plims)

    def lnlike(theta, x, y, yerr):
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
            gp.compute(x, np.sqrt(theta[3]+yerr**2))
        except (ValueError, np.linalg.LinAlgError):
            return 10e25
        return gp.lnlikelihood(y, quiet=True)

    def neglnlike(theta, x, y, yerr):
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
            gp.compute(x, np.sqrt(theta[3]+yerr**2))
        except (ValueError, np.linalg.LinAlgError):
            return 10e25
        return -gp.lnlikelihood(y, quiet=True)

    def MCMC(theta_init, plims, burnin, run)
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

        ndim, nwalkers = len(theta_init), 32
        p0 = [theta_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
        args = (self.x, self.y, self.yerr, plims)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)

        print("burning in...")
        p0, lp, state = sampler.run_mcmc(p0, burnin)
        sampler.reset()
        print("production run...")
        p0, lp, state = sampler.run_mcmc(p0, run)
        return sampler

    def fit(id, theta_init=np.log([1., 1., 1., 1., 1.]), lower=.7, upper=1.5,
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

        p_init, p_err = corr_run(self.x, self.y, self.yerr)
        print "acf period, err = ", p_init

        # run MCMC
        theta_init[-1] = np.log(p_init)
        plims = np.log([p_init*lower, p_init*upper])
        MCMC(id, theta_init, p_init, plims, burnin=burnin, run=run)

    # Dan Foreman-Mackey's acf function
    def dan_acf(x, axis=0, fast=False):
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
        x = np.atleast_1d(x)
        m = [slice(None), ] * len(x.shape)

        # For computational efficiency, crop the chain to the largest power of
        # two if requested.
        if fast:
            n = int(2**np.floor(np.log2(x.shape[axis])))
            m[axis] = slice(0, n)
            x = x
        else:
            n = x.shape[axis]

        # Compute the FFT and then (from that) the auto-correlation function.
        f = np.fft.fft(x-np.mean(x, axis=axis), n=2*n, axis=axis)
        m[axis] = slice(0, n)
        acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[m].real
        m[axis] = 0
        return acf / acf[m]

    def simple_acf(x, y):
        """
        A simple implementation of the ACF method.
        Takes x and y, returns period, smoothed acf, lags and flags
        """
        # fit and subtract straight line
        AT = np.vstack((x, np.ones_like(x)))
        ATA = np.dot(AT, AT.T)
        m, b = np.linalg.solve(ATA, np.dot(AT, y))
        y -= m*x + b

        # perform acf
        acf = dan_acf(y)

        # smooth with Gaussian kernel convolution
        Gaussian = lambda x, sig: 1./(2*np.pi*sig**.5) * \
                     np.exp(-0.5*(x**2)/(sig**2))
        conv_func = Gaussian(np.arange(-28,28,1.), 9.)
        acf_smooth = np.convolve(acf, conv_func, mode='same')

        # create 'lags' array
        gap_days = 0.02043365
        lags = np.arange(len(acf))*gap_days

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

        # flag tells you whether you might believe the ACF results!
        flag = 1  # 1 is good, 0 is bad
        if h < 0:
            flag = 0

        return period, acf_smooth, lags, flag
