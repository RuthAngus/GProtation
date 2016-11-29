from __future__ import print_function, division

import os
import numpy as np
import pandas as pd

from pkg_resources import resource_filename

from .config import AIGRAIN_DIR
from .lc import LightCurve

from .summary import corner_plot

qtr_times = pd.read_table(resource_filename('gprot', 'data/qStartStop.txt'), 
                          delim_whitespace=True, index_col=0)

def get_true_period(i):
    lc = AigrainLightCurve(i)
    P1, P2 = lc.sim_params.P_MIN, lc.sim_params.P_MAX
    return (P1, P2)

class AigrainLightCurve(LightCurve):
    subdir = 'final'
    def __init__(self, i, ndays=None, sub=40, rng=None, nsigma=5, 
                 quarters=None, **kwargs):
        self.i = i

        sid = str(int(i)).zfill(4)
        x, y = np.genfromtxt(os.path.join(AIGRAIN_DIR, self.subdir,
                             "lightcurve_{0}.txt".format(sid))).T
        yerr = np.ones(len(y)) * 1e-5

        if quarters is None:
            self.quarters = None
        else:
            try:
                self.quarters = sorted(quarters)
            except TypeError:
                self.quarters = [quarters]

        if quarters is not None:
            m = np.zeros(x.shape).astype(bool)
            for qtr in self.quarters:
                t0, t1 = qtr_times.ix[qtr, ['tstart', 'tstop']]
                m |= (x >= t0) & (x <= t1)
            x = x[m]
            y = y[m]
            yerr = yerr[m]

        super(AigrainLightCurve, self).__init__(x, y - 1, yerr, **kwargs)

        # Restrict range if desired
        if rng is not None:
            self.restrict_range(rng)
        elif ndays is not None:
            self.restrict_range((0, ndays))

        self.subsample(sub)
        if nsigma is not None:
            self.sigma_clip(nsigma)

        self._sim_params = None

    @property
    def name(self):
        name = str(self.i)
        if self.quarters is not None:
            for q in self.quarters:
                name += '-Q{}'.format(q)

        return name

    def sigma_clip(self, nsigma=5):
        super(AigrainLightCurve, self).sigma_clip(nsigma)

    @property
    def sim_params(self):
        if self._sim_params is None:
            self._sim_params = AigrainTruths().df.ix[self.i]
        return self._sim_params

    def corner_plot(self, samples, **kwargs):
        P1, P2 = self.sim_params.P_MIN, self.sim_params.P_MAX
        return corner_plot(samples, true_period=(np.log(P1), np.log(P2)))

    def subsample(self, *args, **kwargs):
        if 'seed' not in kwargs:
            kwargs['seed'] = self.i
        super(AigrainLightCurve, self).subsample(*args, **kwargs)

    def _make_chunks(self, *args, **kwargs):
        self._split_quarters()

    def _split_quarters(self):
        self._x_list = []
        self._y_list = []
        self._yerr_list = []

        for qtr, (t0, t1) in qtr_times.iterrows():
            if self.quarters is not None and qtr not in self.quarters:
                continue
            m = (self._x >= t0) & (self._x <= t1)
            if m.sum()==0:
                continue
            self._x_list.append(self._x[m])
            self._y_list.append(self._y[m])
            self._yerr_list.append(self._yerr[m])


class NoiseFreeAigrainLightCurve(AigrainLightCurve):
    subdir = 'noise_free'

class AigrainTruths(object):
    filename = os.path.join(AIGRAIN_DIR, 'par', 'final_table.txt')

    def __init__(self):
        self._df = None

    @property
    def df(self):
        if self._df is None:
            self._df = pd.read_table(self.filename, delim_whitespace=True)
        return self._df
