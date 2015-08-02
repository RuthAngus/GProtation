import numpy as np

def bin_data(npts):
"""
A function for binning your data.
Binning is sinning, of course, but if you want to get things
set up quickly this can be very helpful!
It takes your data: x, y, yerr
npts (int) is the number of points per bin.
"""
    mod, nbins = len(x) % npts, len(x) / npts
    if mod != 0:
        x, y, yerr = x[:-mod], y[:-mod], yerr[:-mod]
    xb, yb, yerrb = [np.zeros(nbins) for i in range(3)]
    for i in range(npts):
        xb += x[::npts]
        yb += y[::npts]
        yerrb += yerr[::npts]**2
        x, y, yerr = x[1:], y[1:], yerr[1:]
    return xb/npts, yb/npts, yerrb**.5/npts
