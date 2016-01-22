from __future__ import print_function
import numpy as np
import glob
from kepler_data import load_kepler_data

def combine(fname, d):
    """
    This functions takes the name of a file that contains a list of kepler
    ids and combines the quarters of data together for each star, then
    saves the light curves as a .txt with time, flux, flux_err
    """

    ids = np.genfromtxt(fname, dtype=int).T
    for j, i in enumerate(ids):
        id = str(i)
        print(id)
        fits = np.sort(glob.glob("{0}/{1}/*llc.fits".format(d, id.zfill(9))))
        x, y, yerr = load_kepler_data(fits)
        data = np.vstack((x, y, yerr))
        np.savetxt("data/{0}_lc.txt".format(id), data.T)

if __name__ == "__main__":
        d = "/home/angusr/.kplr/data/lightcurves"
        fname = "data/quiet_kepler_ids.txt"
        combine(fname, d)
