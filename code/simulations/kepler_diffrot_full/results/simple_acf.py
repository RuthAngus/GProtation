from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import emcee
import glob

# dan's acf function
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

def simple_acf(id, x, y, interval, fn, plot):

    # fit and subtract straight line
    AT = np.vstack((x, np.ones_like(x)))
    ATA = np.dot(AT, AT.T)
    m, b = np.linalg.solve(ATA, np.dot(AT, y))
    y -= m*x + b

    # perform acf
    half_acf = dan_acf(y)

    # reflect acf about the zero time axis
    acf = np.concatenate((half_acf[::-1], half_acf))

    # smooth with Gaussian kernel convolution
    Gaussian = lambda x, sig: 1./(2*np.pi*sig**.5) * \
                 np.exp(-0.5*(x**2)/(sig**2))
    conv_func = Gaussian(np.arange(-28,28,1.), 9.)
    acf_smooth = np.convolve(acf, conv_func, mode='same')

    # throw away the first half of the acf
    acf_smooth = acf_smooth[len(acf_smooth)/2:]

    # create 'lags' array
    lags = np.arange(len(acf_smooth)) * interval

    # throw away the first quarter of a day
    c = 12  # one quarter of a day
    acf_smooth, acf, lags = acf_smooth[c:], acf[c:], lags[c:]

    # find all the peaks
    peaks = np.array([i for i in range(1, len(lags)-1)
                     if acf_smooth[i-1] < acf_smooth[i] and
                     acf_smooth[i+1] < acf_smooth[i]])

    # find the first and second peaks
    if len(peaks) > 1:  # if there is more than one peak
        if acf_smooth[peaks[0]] > acf_smooth[peaks[1]]:
            period = lags[peaks[0]]
        else: period = lags[peaks[1]]
    else:
        period = lags[peaks[0]]

    if plot:
        make_plot(x, y, acf_smooth, lags, id, fn)

    return period, acf_smooth, lags

def make_plot(x, y, acf_smooth, lags, id, fn):
        # find all the peaks
        peaks = np.array([i for i in range(1, len(lags)-1)
                         if acf_smooth[i-1] < acf_smooth[i] and
                         acf_smooth[i+1] < acf_smooth[i]])

        # find the lag of highest correlation
        m = acf_smooth ==  max(acf_smooth[peaks])
        highest = lags[m]

        # find the first and second peaks
        if acf_smooth[peaks[0]] > acf_smooth[peaks[1]]:
            period = lags[peaks[0]]
        else: period = lags[peaks[1]]
        print(period)

        plt.clf()
        plt.subplot(4, 1, 1)
        plt.plot(x, y, "k.")
        plt.subplot(4, 1, 2)
        plt.plot(x, y, "k.")
        plt.xlim(0, 200)
        plt.subplot(4, 1, 3)
        plt.plot(x, y, "k.")
        plt.xlim(0, 20)
        plt.subplot(4, 1, 4)
        for i in peaks:
            plt.axvline(lags[i], color="r")
        plt.axvline(highest, color="g")
        plt.axvline(period, color="k")
        plt.plot(lags, acf_smooth)
        plt.xlim(0, 100)
        print("saving plot as", "%s/%s_acf.png" % (fn, str(id).zfill(4)))
        plt.savefig("%s/%s_acf" % (fn, str(id).zfill(4)))

if __name__ == "__main__":

    DIR = "."  # edit me!
    fnames = glob.glob("%s/*.dat" % DIR)

    for i, fname in enumerate(fnames[1:]):
        id = fname.split("/")[-1].split("_")[0]  # edit me!
        x, y, _, _ = np.genfromtxt(fname, skip_header=1).T
        yerr = np.ones_like(y) * 1e-5  # FIXME

        period, acf, lags = simple_acf(x, y)
        make_plot(acf, lags, id)
