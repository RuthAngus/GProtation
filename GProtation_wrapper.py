import numpy as np
import matplotlib.pyplot as plt
import glob
from GProtation import MCMC, make_plot

def wrapper(x, y, yerr, ID, DIR):

        # subsample
        s = 10
        x, y, yerr = x[::s], y[::s], yerr[::s]

        # median normalise
        med = np.median(y)
        y /= med
        yerr /= med

#         # try optimising
#         print(ID)
#         p_init = float(raw_input("Enter initial guess for rotation period "))
#         theta = intitialisation(x, y, yerr, p_init, ID, DIR)
#         print(theta)
#         choose = raw_input("Use these parameters? y/n ")
#         if choose == "y": theta = theta
#         else: theta = [1., 1., 1., p_init, 1.]

#         theta = [1., 1., 1., 8., 1.]
        theta = np.log([1e-6, 20.**2, 20, 8., 1e-7])
        sampler = MCMC(np.array(theta), x, y, yerr, ID, DIR)
        make_plot(sampler, ID, DIR)

def kepler():
    D = "/Users/angusr/angusr/data2/Q15_public"  # data directory
    DIR = "/Users/angusr/Python/GProtation/mcmc"  # results directory
    fnames = glob.glob("%s/kplr0081*" % D)

    for fname in fnames:
        x, y, yerr = load_fits(fname)
        l = np.isfinite(x) * np.isfinite(y) * np.isfinite(yerr)
        x, y, yerr = x[l], y[l], yerr[l]
        kid = fname[42:51]
        wrapper(x, y, yerr, fname, DIR)

def K2():
    D = "/Users/angusr/data/K2/c0corcutlcs"  # data directory
    DIR = "/Users/angusr/Python/GProtation/mcmc"  # results directory
    fnames = glob.glob("%s/ep*.csv" % D)

    for fname in fnames:
        x, y, empty = np.genfromtxt(fname, skip_header=1, delimiter=",").T
        yerr = np.ones_like(y)*1e-5
        ID = fname[36:45]
        wrapper(x, y, yerr, ID, DIR)

if __name__ == "__main__":
    K2()
