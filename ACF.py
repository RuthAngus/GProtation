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
        n = np.where(diff > 2*mdiff)[0]

        # big arrays
        xb = x[:n[0]]
        yb = y[:n[0]]

        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(x, y, "k.")

        for i in n:
                xb = np.append(xb, x[i:i+1])
                yb = np.append(yb, y[i:i+1])

                # fill the gap
                xgap = x[i:i+2]
                ygap = y[i:i+2]

                f = spi.interp1d(xgap, ygap)
                xnew = np.arange(xgap[0], xgap[-1], t_int)
                yerrnew = np.ones_like(xnew)*merr
                ynew = f(xnew) + np.random.randn(len(xnew))*merr  # add noise

                xb = np.append(xb, xnew)
                yb = np.append(yb, ynew)

        plt.subplot(2, 1, 2)
        plt.plot(xb, yb, "k.")
        plt.savefig("test")
        raw_input('enter')

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
        l = np.isfinite(x)*np.isfinite(y)*np.isfinite(yerr)*(q==0)
        x, y, yerr = x[l], y[l], yerr[l]

        interpolate(x, y, yerr)
