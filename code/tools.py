import numpy as np

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
