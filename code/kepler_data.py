from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pyfits
import glob

def load_kepler_data(fnames):
    hdulist = pyfits.open(fnames[0])
    t = hdulist[1].data
    time = t["TIME"]
    flux = t["PDCSAP_FLUX"]
    flux_err = t["PDCSAP_FLUX_ERR"]
    q = t["SAP_QUALITY"]
    m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err) * \
            (q == 0)
    x = time[m]
    med = np.median(flux[m])
    y = flux[m]/med - 1
    yerr = flux_err[m]/med
    for fname in fnames[1:]:
       hdulist = pyfits.open(fname)
       t = hdulist[1].data
       time = t["TIME"]
       flux = t["PDCSAP_FLUX"]
       flux_err = t["PDCSAP_FLUX_ERR"]
       q = t["SAP_QUALITY"]
       m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err) * \
               (q == 0)
       x = np.concatenate((x, time[m]))
       med = np.median(flux[m])
       y = np.concatenate((y, flux[m]/med - 1))
       yerr = np.concatenate((yerr, flux_err[m]/med))
    return x, y, yerr

# load Kepler IDs
data = np.genfromtxt("data/garcia.txt", skip_header=1).T
kids = data[0]
id = kids[0]
path = "/.kplr/data/lightcurves/{0}/*fits".format(str(int(id)).zfill(9))
fnames = glob.glob(path)
x, y, yerr = load_kepler_data(path)

plt.clf()
plt.plot(x, y, "k.")
plt.savefig("test")
