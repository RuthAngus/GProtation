from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LightCurve(object):
    def __init__(self, x, y, yerr, name='', chunksize=200):
        self._x = x.copy()
        self._y = y.copy()
        self._yerr = yerr.copy()

        self.chunksize = chunksize

        self._name = name

        self._x_list = None
        self._y_list = None
        self._yerr_list = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._y = val

    @property
    def yerr(self):
        return self._yerr

    @yerr.setter
    def yerr(self, val):
        self._yerr = val

    @property
    def name(self):
        return self._name

    @property
    def df(self):
        return pd.DataFrame({'x':self.x, 'y':self.y, 'yerr':self.yerr})

    def _make_chunks(self, chunksize=None):
        if chunksize is None:
            if self.chunksize is None:
                return
            chunksize = self.chunksize
        N = len(self.x) // chunksize
        self.chunksize = chunksize
        self._x_list = np.array_split(self.x, N)
        self._y_list = np.array_split(self.y, N)
        self._yerr_list = np.array_split(self.yerr, N)

    @property
    def x_list(self):
        if self._x_list is None:
            self._make_chunks()
        return self._x_list

    @property
    def y_list(self):
        if self._y_list is None:
            self._make_chunks()
        return self._y_list

    @property
    def yerr_list(self):
        if self._yerr_list is None:
            self._make_chunks()
        return self._yerr_list

    @property
    def is_split(self):
        return self.x_list is not None

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

    def subsample(self, sub, seed=None):
        """Random subsampling
        """
        if sub is None:
            return
        N = len(self.x)
        np.random.seed(seed)
        inds = np.sort(np.random.choice(N, N//sub, replace=False))
        self.x = self.x[inds]
        self.y = self.y[inds]
        self.yerr = self.yerr[inds]

    def polyflat(self, order=3):
        p = np.polyfit(self.x, self.y, order)
        self.y -= np.polyval(p, self.x)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()

        ax.plot(self.x, self.y, **kwargs)
        ax.set_xlabel('Time [d]')
        ax.set_ylabel('Flux')
        ax.set_title(self.name)
        return fig
