from __future__ import print_function
import numpy as np
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import emcee
import triangle
import h5py

class gisela(object)

    def __init__(self):

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

    def MCMC(theta_init, x, y, yerr, plims, burnin, run)
        """
        This function runs emcee in order to sample the posterior of your
        hyperparameters or find the maxima, etc.

        It takes an initial guess for your parameters (theta_init).
        These must be logarithmic.
        It takes your data: x, y, yerr.
        plims are your upper and lower prior bounds on the period.
        burnin (integer) is the number of steps during burn-in.
        run (integer) is the number of steps for the full run.

        In practise I find that 32 walkers are fine, so I've fixed this
        number.
        This function returns the emcee sampler object.
        See dfm.io/emcee for emcee documentation.
        """

        ndim, nwalkers = len(theta_init), 32
        p0 = [theta_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
        args = (x, y, yerr, plims)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)

        print("burning in...")
        p0, lp, state = sampler.run_mcmc(p0, burnin)
        sampler.reset()
        print("production run...")
        p0, lp, state = sampler.run_mcmc(p0, run)

        return sampler

    def bin_data(x, y, yerr, npts):
    """
    A function for binning your data.
    Binning is sinning, of course, but if you want to get things
    set up quickly this can be very helpful!
    It takes your data: x, y, yerr
    npts (int) is the number of points per bin.
    """
        mod, nbins = len(x) % npts, len(x) / npts
        if mod != 0:
            x, y, yerr = x[:-mod], y[:-mod], yerr[:-mod]
        xb, yb, yerrb = [np.zeros(nbins) for i in range(3)]
        for i in range(npts):
            xb += x[::npts]
            yb += y[::npts]
            yerrb += yerr[::npts]**2
            x, y, yerr = x[1:], y[1:], yerr[1:]
        return xb/npts, yb/npts, yerrb**.5/npts

    def fit(x, y, yerr, id, burnin=500, run=1500, npts=48, cutoff=100,
            sine_kernel=False, acf=False, runMCMC=True,
            plot=False):
        """
        Takes x, y, yerr and initial guesses and priors for period, bins your
        data sensibly, truncates it for speed and does the full GP MCMC.
        x, y, yerr (arrays): your data.
        id (str or int): the name of your object
        p_init (tuple): upper and lower limits for your period prior
        burnin (int): number of burnin steps
        run (int): number of steps for the full run
        npts (int): number of points per bin
        """

        # measure ACF period
        if acf:
            corr_run(x, y, yerr, id, "/Users/angusr/Python/GProtation/code")

        if sine_kernel:
            print "sine kernel"
            theta_init = [1., 1., 1., 1., p_init]
            print theta_init
            from GProtation import MCMC, make_plot
        else:
            print "cosine kernel"
            theta_init = [1e-2, 1., 1e-2, p_init]
            print "theta_init = ", np.log(theta_init)
            from GProtation_cosine import MCMC, make_plot

        xb, yb, yerrb = bin_data(x, y, yerr, npts)  # bin data
        m = xb < cutoff  # truncate

        theta_init = np.log(theta_init)
        DIR = "cosine"
        if sine_kernel:
            DIR = "sine"

        print theta_init
        if runMCMC:
            sampler = MCMC(theta_init, xb[m], yb[m], yerrb[m], plims, burnin, run,
                           id, DIR)

        # make various plots
        if plot:
            with h5py.File("%s/%s_samples.h5" % (DIR, str(int(id)).zfill(4)),
                           "r") as f:
                samples = f["samples"][...]
            m = x < cutoff
            mcmc_result = make_plot(samples, x[m], y[m], yerr[m], id, DIR,
                                    traces=False, triangle=False, prediction=True)

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
