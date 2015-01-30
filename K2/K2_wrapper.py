import numpy as np
from amy_acf import corr_run
import glob
import subprocess

DIR = "/Users/angusr/data/K2/c0corcutlcs"
fnames = glob.glob("%s/ep*.csv" % DIR)
print len(fnames)
for i, fname in enumerate(fnames[651:]):
    ID = fname[36:45]
    x, y, empty = np.genfromtxt(fname, skip_header=1, delimiter=",").T
    corr_run(x, y, ID)
