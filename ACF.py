# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import emcee
import scipy.interpolate as spi
import pyfits

def interpolate(x, y, yerr):
    # mean err
    merr = np.mean(yerr)

    # find gaps
    diff = x[1:] - x[:-1]
    mdiff = min(diff)  # sp.stats.mstats.mode(diff)
    l = diff > 2*mdiff
    m = l==False
    t_int = np.mean(diff[m])

    # these are the gaps
    n = np.where(diff > 1.1*mdiff)[0]
    m = np.where(diff < 1.1*mdiff)[0]

    # big arrays
    xb = x[:n[0]]
    yb = y[:n[0]]

    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(x, y, "k.")

    for i in range(len(n)-1):

            # fill the gap
            xgap = x[n[i]:n[i+1]+1]
            ygap = y[n[i]:n[i+1]+1]
#             if len(xgap) < 2:
#                 print xgap, n[i], n[i+1], x[n[i+1]]-x[n[i]]
#                 xgap = x[n[i]:n[i+1]+1]
#                 ygap = y[n[i]:n[i+1]+1]

            f = spi.interp1d(xgap, ygap)
            xnew = np.arange(xgap[0], xgap[-1], t_int)
            print len(xnew)
            yerrnew = np.ones_like(xnew)*merr
            ynew = f(xnew) + np.random.randn(len(xnew))*merr  # add noise

            if len(xgap) < 2:
                xb = np.append(xb, xnew[:-1])
                yb = np.append(yb, ynew[:-1])
            else:
                xb = np.append(xb, xnew)
                yb = np.append(yb, ynew)

    plt.subplot(2, 1, 2)
    plt.plot(xb, yb, "k.")
    plt.savefig("test2")

    diff = xb[1:] - xb[:-1]
    print len(n), len(diff[diff<0.01]), len(diff[diff>0.03])
    print len(x), len(np.unique(x))

def kplr_interp(x, y, yerr):
    t = np.arange(len(y))
    l = np.isfinite(y)*np.isfinite(x)
    y[~l] = np.interp(t[~l], t[l], y[l])
    x[~l] = np.interp(t[~l], t[l], x[l])
    yerr[~l] = np.median(yerr[l])
    return x, y, yerr

# calculuate autocorrelation (assume evenly spaced)
def acf(x, y):
        LC = np.vstack((x, y))
        return emcee.autocorr.function(LC, axis=0)

if __name__ == "__main__":

        DIR = "/Users/angusr/.kplr/data/lightcurves/010355856"
        hdulist = pyfits.open("%s/kplr010355856-2013131215648_llc.fits" % DIR)
        tbdata = hdulist[1].data
        x = tbdata["TIME"]
        y = tbdata["PDCSAP_FLUX"]
        yerr = tbdata["PDCSAP_FLUX_ERR"]
        q = tbdata["SAP_QUALITY"]

        plt.clf()
        plt.subplot(2, 1, 1)
        plt.errorbar(x, y, yerr=yerr, fmt="k.")

        x, y, yerr = kplr_interp(x, y, yerr)
        plt.subplot(2, 1, 2)
        plt.errorbar(x, y, yerr=yerr, fmt="k.")
        plt.savefig("test")
