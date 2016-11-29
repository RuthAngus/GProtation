from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pkg_resources import resource_filename

qtr_times = pd.read_table(resource_filename('gprot', 'data/qStartStop.txt'), 
                          delim_whitespace=True, index_col=0)

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

        self._x_full = x.copy()
        self._y_full = y.copy()
        self._yerr_full = yerr.copy()

    @property
    def name(self):
        return self._name

    @property
    def df(self):
        return pd.DataFrame({'x':self.x, 'y':self.y, 'yerr':self.yerr})

    def multi_split_quarters(self, qtrs, subs, seed=None):
        self._x_list = []
        self._y_list = []
        self._yerr_list = []
        np.random.seed(seed)
        for (qtr, (t0, t1)), sub in zip(qtr_times.iterrows(), subs):
            m = (self.x_full >= t0) & (self.x_full <= t1)
            N = int(m.sum())
            if N==0:
                continue
            inds = np.sort(np.random.choice(N, int(N//sub), replace=False))
            self._x_list.append(self.x_full[m][inds])
            self._y_list.append(self.y_full[m][inds])
            self._yerr_list.append(self.yerr_full[m][inds])

        self.x = np.concatenate(self._x_list)
        self.y = np.concatenate(self._y_list)
        self.yerr = np.concatenate(self._yerr_list)

    def _split_quarters(self):
        if not hasattr(self, 'quarters'):
            raise AttributeError('Cannot split quarters if quarters not defined.')
        self._x_list = []
        self._y_list = []
        self._yerr_list = []

        for qtr, (t0, t1) in qtr_times.iterrows():
            if self.quarters is not None and qtr not in self.quarters:
                continue
            m = (self.x >= t0) & (self.x <= t1)
            if m.sum()==0:
                continue
            self._x_list.append(self.x[m])
            self._y_list.append(self.y[m])
            self._yerr_list.append(self.yerr[m])

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

    def _get_data(self):
        pass

    @property
    def x(self):
        if self._x is None:
            self._get_data()
        return self._x

    @property
    def y(self):
        if self._y is None:
            self._get_data()
        return self._y

    @property
    def yerr(self):
        if self._yerr is None:
            self._get_data()
        return self._yerr

    @property
    def x_full(self):
        if self._x_full is None:
            self._get_data()
        return self._x_full

    @property
    def y_full(self):
        if self._y_full is None:
            self._get_data()
        return self._y_full

    @property
    def yerr_full(self):
        if self._yerr_full is None:
            self._get_data()
        return self._yerr_full

    @x.setter
    def x(self, val):
        self._x = val
        
    @y.setter
    def y(self, val):
        self._y = val

    @yerr.setter
    def yerr(self, val):
        self._yerr = val

