from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

from scipy.interpolate import UnivariateSpline
from scipy.signal import boxcar
from scipy.ndimage.filters import convolve
from scipy.optimize import minimize

from collections import OrderedDict
from pkg_resources import resource_filename

from .filter import sigma_clip, bandpass_filter
from .plots import tableau20
from .acf import acf, peakdetect

qtr_times = pd.read_table(resource_filename('gprot', 'data/qStartStop.txt'), 
                          delim_whitespace=True, index_col=0)

class LightCurve(object):
    def __init__(self, x, y, yerr, name=None, chunksize=200, sub=None):
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

        self.sub = sub
        self.subsample(sub)

    @property
    def name(self):
        if self._name is None:
            self._name = 'lc'
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def df(self):
        return pd.DataFrame({'x':self.x, 'y':self.y, 'yerr':self.yerr})

    def bandpass_filter(self, pmin=0.5, pmax=100, cadence=None, edge=2000,
                        zero_fill=False):
        """Applies a Butterworth bandpass filter to data

        Replaces lightcurve data with new filtered, edge-cropped data.
        """
        if cadence is None:
            try:
                cadence = self.cadence
            except AttributeError:
                pass

        x, y, yerr = bandpass_filter(self._x_full,
                                     self._y_full,
                                     self._yerr_full, zero_fill=zero_fill,
                                     pmin=pmin, pmax=pmax)
        x = x[edge:-edge]
        y = y[edge:-edge]
        yerr = yerr[edge:-edge]

        self._x = x.copy()
        self._y = y.copy()
        self._yerr = yerr.copy()

        self._x_full = x.copy()
        self._y_full = y.copy()
        self._yerr_full = yerr.copy()

        self._x_list = None
        self._y_list = None
        self._yerr_list = None

        if self.sub is not None:
            self.subsample(self.sub)

    def acf(self, pmin=0.1, pmax=100, filter=True, smooth=None):
        """Filters with pmax = pmax, then returns ACF up to lag=2*pmax
        """
        if filter:
            if self._x_full is None:
                self._get_data()

            x, y, yerr = bandpass_filter(self._x_full,
                                         self._y_full,
                                         self._yerr_full, zero_fill=True,
                                         pmin=pmin, pmax=pmax)
        else:
            x, y = self.x, self.y

        lags, ac = acf(x, y, maxlag=2*pmax)

        if smooth is not None:
            cadence = np.median(np.diff(lags))
            Nbox = smooth / cadence 
            if Nbox >= 3:
                ac = convolve(ac, boxcar(Nbox)/float(Nbox), mode='reflect')

        return lags, ac

    def plot_acf(self, truth=None, **kwargs):
        lags, ac = self.acf(**kwargs)

        fig, ax = plt.subplots(1,1)
        ax.plot(lags, ac, 'k')
        if truth is not None:
            ax.axvline(truth, c='r', ls=':')

        return fig

    def acf_prot(self, pmin=0.1, pmax=100, delta=0.02, lookahead=30,
                 peak_to_trough=True, maxpeaks=1, plot=False, ax=None,
                 fig_kwargs=None, savefig_filename=None):
        """Returns best guess of prot from ACF, and height of peak

        Just pick first peak.
        """
        lags, ac = self.acf(pmin=pmin, pmax=pmax, smooth=pmax/10)

        # make sure lookahead isn't too long if pmax is small
        lookahead = min(lookahead, pmax)

        n_maxes = 0
        while n_maxes == 0:
            maxes, mins = peakdetect(ac, lags, delta=delta, lookahead=lookahead)

            # First max only counts if it's after a min.
            try:
                if mins[0][0] > maxes[0][0]:
                    maxes.pop(0)
            except IndexError:
                pass

            n_maxes = len(maxes)
            if n_maxes == 0:
                delta /= 2

        maxheight = -np.inf
        pbest = np.nan
        for i, ((xhi, yhi), (xlo, ylo)) in enumerate(zip(maxes, mins)):
            if peak_to_trough:
                # Calculate mean(peak-to-trough) height
                h1 = yhi - ylo
                try:
                    h2 = yhi - mins[i+1][1]
                    height = (h1+h2)/2.
                except IndexError:
                    height = h1
            else:
                height = yhi
                print(i, height)

            if height > maxheight:
                pbest = xhi
                maxheight = height
            if i == maxpeaks-1:
                break

        # Evaluate quality by fitting exp*sin
        x, y = lags, ac

        def fn(x, A, tau, T):
            return A*np.exp(-x/tau)*np.cos(2*np.pi*x/T)

        def chisq(p):
            A, tau = p

            mod = fn(x, A, tau, pbest)
            return ((mod - y)**2).sum()

        fit = minimize(chisq, [1., pbest*2])

        tau = fit.x[1]

        # Bigger is better. len(lags) is basically proportional to pmax
        quality =  1./ (fit.fun / len(lags) / maxheight)
        quality *= tau/pbest # enhance quality for long decay timescales.

        if ax is not None:
            plot = True
        if plot:
            if ax is None:
                fig, ax = plt.subplots(1,1)
            else:
                fig = ax.get_figure()

            if fig_kwargs is None:
                fig_kwargs = dict(color='k')

            ax.plot(lags, ac, **fig_kwargs)
            if np.isfinite(pbest):
                ax.axvline(pbest, ls=':', color='r')

                ax.plot(lags, fn(lags, fit.x[0], fit.x[1], pbest))

            ax.annotate('P={:.2f}\ntau={:.2f}\nQ={:.1f}'.format(pbest, tau, quality), 
                        xy=(0.8,0.9), xycoords='axes fraction', ha='left', va='top')

        if plot:
            if savefig_filename:
                fig.savefig(savefig_filename)
                return pbest, maxheight, tau, quality
            else:
                return pbest, maxheight, tau, quality, fig
        else:
            return pbest, maxheight, tau, quality

    def best_sublc(self, ndays, npoints=600, chunksize=300,
                    flat_order=3, **kwargs):
        """Returns new sub-LightCurve, choosing ndays with maximum RMS variation 
        """
        x_full = self.x_full
        y_full = self.y_full

        N = len(x_full)
        cadence = np.median(x_full[1:] - x_full[:-1])
        window = int(ndays / cadence)
        stepsize = window//50
        i1 = 0
        i2 = i1 + window
        max_std = 0
        max_i1 = None
        max_i2 = None
        while i2 < N:
            x = x_full[i1:i2].copy()
            y = y_full[i1:i2].copy()
            x, y, _ = sigma_clip(x, y, y, 5)
            p = np.polyfit(x, y, flat_order)
            y -= np.polyval(p, x)
            std = np.std(y)
            if std > max_std:
                max_i1 = i1
                max_i2 = i2
                max_std = std
            i1 += stepsize
            i2 += stepsize

        x, y, yerr = (x_full[max_i1:max_i2], 
                      y_full[max_i1:max_i2], 
                      self.yerr_full[max_i1:max_i2])

        newname = self.name + '_{:.0f}d'.format(ndays)
        if 'sub' not in kwargs:
            kwargs['sub'] = window//npoints

        return LightCurve(x, y, yerr, chunksize=chunksize,
                          name=newname, **kwargs)        

    def make_best_chunks(self, ndays=[800, 200, 50], seed=None, **kwargs):
        if not hasattr(ndays, '__iter__'):
            ndays = [ndays]

        self._x_list = []
        self._y_list = []
        self._yerr_list = []

        np.random.seed(seed)
        for nd in ndays:
            lc = self.best_sublc(nd, **kwargs)
            self._x_list += lc.x_list
            self._y_list += lc.y_list
            self._yerr_list += lc.yerr_list

    def multi_split_quarters(self, qtrs, subs, seed=None):
        self._x_list = []
        self._y_list = []
        self._yerr_list = []
        np.random.seed(seed)
        for qtr, sub in zip(qtrs, subs):
            t0, t1 = qtr_times.ix[qtr]
            m = (self.x_full >= t0) & (self.x_full <= t1)
            N = int(m.sum())
            # print(N, sub, N//sub)
            if N==0:
                continue
            inds = np.sort(np.random.choice(N, int(N//sub), replace=False))
            self._x_list.append(self.x_full[m][inds])
            self._y_list.append(self.y_full[m][inds])
            self._yerr_list.append(self.yerr_full[m][inds])

        self.x = np.concatenate(self._x_list)
        self.y = np.concatenate(self._y_list)
        self.yerr = np.concatenate(self._yerr_list)

    def multi_split_quarters_rms(self, subs=None,
                                 seed=None):
        d = self.qtr_rms()
        if subs is None:
            subs = np.ones(len(d))*40
            subs[:4] = [5, 10, 20, 30]
        qtrs = np.array(d.keys())[np.argsort(d.values())[::-1]][:len(subs)]
        self.multi_split_quarters(qtrs, subs, seed=seed)

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
        nall = len(self.x)
        N = len(self.x) // chunksize 
        # Ensure chunks *no larger* than chunksize.
        if not nall % chunksize == 0:
            N += 1
        self.chunksize = chunksize
        self._x_list = np.array_split(self.x, N)
        self._y_list = np.array_split(self.y, N)
        self._yerr_list = np.array_split(self.yerr, N)

    def chunk_rms(self, t0, t1, nsigma=5):
        """Returns rms flux variability between t0 and t1

        """
        m = (self.x_full >= t0) & (self.x_full <= t1)
        if m.sum()==0:
            return np.nan
        x = self.x_full[m].copy()
        y = self.y_full[m].copy()
        yerr = self.yerr_full[m].copy()

        x, y, yerr = sigma_clip(x, y, yerr, nsigma)

        p = np.polyfit(x, y, 1)
        y -= np.polyval(p, x)

        med = np.median(y)
        std = (sum((med - y)**2)/float(len(y)))**.5

        return std

    def qtr_rms(self, nsigma=5):
        rms_all = {}
        for qtr, (t0, t1) in qtr_times.iterrows():
            if self.quarters is not None and qtr not in self.quarters:
                continue
            rms = self.chunk_rms(t0, t1, nsigma=nsigma)
            if np.isfinite(rms):
                rms_all[qtr] = rms
        return rms_all

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
        self.x, self.y, self.yerr = sigma_clip(self.x, self.y, self.yerr, nsigma)

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

        if self.x_list is not None:
            ax.set_prop_cycle(cycler('color', tableau20[::2]))            
            for x, y in zip(self.x_list, self.y_list):
                ax.plot(x, y, **kwargs)
        else:
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

