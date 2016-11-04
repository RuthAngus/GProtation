from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from .filter import sigma_clip

class LightCurve(object):
    def __init__(self, x, y, yerr):
        self.x = x.copy()
        self.y = y.copy()
        self.yerr = yerr.copy()

        self._x_raw = x.copy()
        self._y_raw = y.copy()
        self._yerr_raw = yerr.copy()

    def sigma_clip(self, nsigma):
        med = np.median(self.y)
        std = (sum((med - self.y)**2)/float(len(self.y)))**.5
        m = np.abs(self.y - med) > (nsigma * std)

        self.x = self.x[~m]
        self.y = self.y[~m]
        self.yerr = self.yerr[~m]

    def restrict_range(self, rng):
        m = (self.x > rng[0]) & (self.x < rng[1])
        self.x = self.x[m]
        self.y = self.y[m]
        self.yerr = self.yerr[m]

    def subsample(self, sub):
        self.x = self.x[::sub]
        self.y = self.y[::sub]
        self.yerr = self.yerr[::sub]

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()

        ax.plot(self.x, self.y, **kwargs)
        ax.set_xlabel('Time [d]')
        ax.set_ylabel('Flux')
