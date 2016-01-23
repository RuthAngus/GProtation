from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from kepler_data import load_kepler_data
import glob

# ids = np.genfromtxt("data/NGC6819_members.txt", dtype=int).T
ids = np.genfromtxt("data/astero_ids.txt", dtype=int).T
var = np.zeros(len(ids))
for j, i in enumerate(ids):
    id = str(i)
    print(id, j, "of", len(ids))
    d = "/home/angusr/.kplr/data/lightcurves"
    fnames = np.sort(glob.glob("{0}/{1}/*llc.fits".format(d, id.zfill(9))))
    x, y, yerr = load_kepler_data(fnames)
    var[j] = np.var(y)

print(var)
print(min(var), max(var))

plt.clf()
plt.hist(np.log10(var))
# plt.savefig("var_NGC6819")
plt.savefig("var_astero")
