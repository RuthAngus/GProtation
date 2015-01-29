import numpy as np
from ACF import load_fits
from amy_acf import corr_run
import glob
import subprocess

def wrap(ID):
    D = "/Users/angusr/angusr/data2/Q15_public"
    fname = "%s/kplr%s-2013011073258_llc.fits" % (D, ID.zfill(9))
    x, y, yerr = load_fits(fname)
    l = np.isfinite(x) * np.isfinite(y) * np.isfinite(yerr)
    x, y, yerr = x[l], y[l], yerr[l]
    corr_run(x, y, ID)

if __name__ == "__main__":
    D = "/Users/angusr/angusr/data2/Q15_public"
    fnames = glob.glob("%s/kplr0081*" % D)

    for fname in fnames:
        x, y, yerr = load_fits(fname)
        l = np.isfinite(x) * np.isfinite(y) * np.isfinite(yerr)
        x, y, yerr = x[l], y[l], yerr[l]
        id_list = fname[42:51]
        print id_list
        corr_run(x, y, id_list)
