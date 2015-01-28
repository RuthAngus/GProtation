# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import emcee
import scipy.interpolate as spi
import pyfits
import glob

def kplr_interp(x, y, yerr):
    t = np.arange(len(y))
    l = np.isfinite(y)*np.isfinite(x)
    y[~l] = np.interp(t[~l], t[l], y[l])
    x[~l] = np.interp(t[~l], t[l], x[l])
    yerr[~l] = np.median(yerr[l])
    return x, y, yerr

# calculuate autocorrelation (assume evenly spaced)
def acf(y):
    return emcee.autocorr.function(y)

# find all the peaks
def peaky(x, y):
    xp, yp = [], []
    for i in range(len(y)-2):
        if y[i+2] < y[i+1] and y[i] < y[i+1]:
            xp.append(x[i+1])
            yp.append(y[i+1])
    return np.array(xp), np.array(yp)

def peak_select(xp, yp):
    l = xp > xp[0]+1  # exclude region after 1st peak
    xn = xp[l]
    yn = yp[l]
    if yn[1] > yn[0]:  # choose 2nd peak if it's higher
        return xn[1:], yn[1:], xn[1]
    return xn, yn, xn[0]

def find_period(kid, x, y, yerr):

        # interpolate
        x, y, yerr = kplr_interp(x, y, yerr)

        # find peaks
        dt = np.median(np.diff(x))
        f = acf(y)
        xp, yp = peaky(dt*np.arange(len(f)), f)

        # find period peak
        xp, yp, p = peak_select(xp, yp)

        # plot
        plt.clf()
        plt.plot(dt*np.arange(len(f)), f)
        plt.axvline(p, color="r")
        plt.savefig("%s_acf" % kid.zfill(9))

        return p

def load_fits(fname):
        hdulist = pyfits.open(fname)
        tbdata = hdulist[1].data
        x = tbdata["TIME"]
        y = tbdata["PDCSAP_FLUX"]
        yerr = tbdata["PDCSAP_FLUX_ERR"]
        q = tbdata["SAP_QUALITY"]
        return x, y, yerr

if __name__ == "__main__":

        kid = "10355856"
        DIR = "/Users/angusr/.kplr/data/lightcurves/%s" % kid.zfill(9)
        fname = "%s/kplr%s-2013131215648_llc.fits" % (DIR, kid.zfill(9))
        x, y, yerr = load_fits(fname)
        p = find_period(kid, x, y, yerr)
        print p

        DIR = "/Users/angusr/angusr/data2/Q15_public"
        fnames = glob.glob("%s/kplr0081*" % DIR)

        for fname in fnames:
            x, y, yerr = load_fits(fname)
            kid = fname[42:51]
            p = find_period(kid, x, y, yerr)
            print kid, p
