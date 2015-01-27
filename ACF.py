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

        # big arrays
        xb = np.zeros(int((x[-1]-x[0])/t_int))
        yb = np.zeros_like(xb)

        # these are the gaps
        n = np.where(diff > 2*mdiff)[0]
        for i in n:
                xb[:i] = x[:i]  # fill in present values
                yb[:i] = y[:i]  # fill in present values

                # fill the gap
                xgap = x[i:i+2]
                ygap = y[i:i+2]

                plt.clf()
                plt.subplot(2, 1, 1)
                plt.plot(xgap, ygap, "k.")

                f = spi.interp1d(xgap, ygap)
                xnew = np.arange(xgap[0], xgap[-1], t_int)
                yerrnew = np.ones_like(xnew)*merr
                ynew = f(xnew) + np.random.randn(len(xnew))*merr  # add noise

                plt.subplot(2, 1, 2)
                plt.plot(xnew, ynew, "k.")
                plt.savefig("test")
                assert 0

                xb[i:i+len(xnew)] = xnew
                yb[i:i+len(xnew)] = ynew

# calculuate autocorrelation (assume evenly spaced)
def acf(x, y):
        LC = np.vstack((x, y))
        return emcee.autocorr.function(LC, axis=0)

if __name__ == "__main__":

        DIR = "/n/home11/rangus/.kplr/data/lightcurves/010355856"
        hdulist = pyfits.open("%s/kplr010355856-2013131215648_llc.fits" % DIR)
        tbdata = hdulist[1].data
        x = tbdata["TIME"]
        y = tbdata["PDCSAP_FLUX"]
        yerr = tbdata["PDCSAP_FLUX_ERR"]
        q = tbdata["SAP_QUALITY"]
        l = np.isfinite(x)*np.isfinite(y)*np.isfinite(yerr)*(q==0)
        x, y, yerr = x[l], y[l], yerr[l]

        interpolate(x, y, yerr)
