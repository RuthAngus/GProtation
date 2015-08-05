import numpy as np
import matplotlib.pyplot as plt
import pyfits
import h5py
from measure_GP_rotation import bin_data
from GProtation import make_plot
import sys
from Kepler_ACF import corr_run

# # setup names and directories
id = str(int(sys.argv[1])).zfill(4)
# sine_kernel, cutoff = True, 100
# DIR = "cosine"
# if sine_kernel:
#     DIR = "sine"

# # load data
x, y = np.genfromtxt("simulations/%s.txt" % str(int(id)).zfill(4)).T
yerr = np.ones_like(y) * 1e-5
#
# try:
#     p_init = np.genfromtxt("simulations/%s_result.txt" % id)
# except:
#     corr_run(x, y, yerr, id,
#              "/Users/angusr/Python/GProtation/code/simulations")
#     p_init = np.genfromtxt("simulations/%s_result.txt" % id)
# print "acf period, err = ", p_init

# # load samples
# with h5py.File("%s/%s_samples.h5" % (DIR, str(int(id)).zfill(4)),
#                "r") as f:
#     samples = f["samples"][...]
# nwalkers, nsteps, ndims = np.shape(samples)
# flat = np.reshape(samples, (nwalkers * nsteps, ndims))
# mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                   zip(*np.percentile(flat, [16, 50, 84], axis=0)))

# bin and truncate data
# npts = int(p_init[0] / 10. * 48)  # 10 points per period
# xb, yb, yerrb = bin_data(x, y, yerr, npts)  # bin data
# m = xb < cutoff  # truncate
m = x < 10

DIR = "sine"
for id in range(100):
    # load samples
    with h5py.File("%s/%s_samples.h5" % (DIR, str(int(id)).zfill(4)),
                   "r") as f:
        samples = f["samples"][...]
    nwalkers, nsteps, ndims = np.shape(samples)
    flat = np.reshape(samples, (nwalkers * nsteps, ndims))
    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(flat, [16, 50, 84], axis=0)))
    # make various plots
    make_plot(samples, x[m], y[m], yerr[m], id, DIR, traces=False, tri=True,
              prediction=False)
