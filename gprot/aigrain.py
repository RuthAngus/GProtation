from __future__ import print_function, division

import os
import numpy as np
import pandas as pd

from .config import AIGRAIN_DIR
from .lc import LightCurve, qtr_times

from .summary import corner_plot

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

    def multi_split_quarters(self):
        if self.quarters is None:
            qtrs = np.arange(10) + 1
        else:
            qtrs = self.quarters

        N = len(qtrs)
        subs = np.ones(len(qtrs))*40
        # have middle be 5, flanked by 10, 20 then 40
        for i, sub in zip(range(3), [5,10,20]):
            subs[N//2 + i] = sub
            subs[N//2 - i] = sub

        super(AigrainLightCurve, self).multi_split_quarters(qtrs, subs, seed=self.i)

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
