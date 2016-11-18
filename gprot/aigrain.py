from __future__ import print_function, division

import os
import numpy as np
import pandas as pd

from .config import AIGRAIN_DIR
from .lc import LightCurve

from .summary import corner_plot

class AigrainLightCurve(LightCurve):
    subdir = 'final'
    def __init__(self, i, ndays=None, sub=40, rng=None, nsigma=5, **kwargs):
        self.i = i

        sid = str(int(i)).zfill(4)
        x, y = np.genfromtxt(os.path.join(AIGRAIN_DIR, self.subdir,
                             "lightcurve_{0}.txt".format(sid))).T
        yerr = np.ones(len(y)) * 1e-5
        super(AigrainLightCurve, self).__init__(x - x[0], y - 1, yerr, name=str(i), **kwargs)

        # Restrict range if desired
        if rng is not None:
            self.restrict_range(rng)
        elif ndays is not None:
            self.restrict_range((0, ndays))

        self.subsample(sub)
        if nsigma is not None:
            self.sigma_clip(nsigma)

        self._sim_params = None

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
