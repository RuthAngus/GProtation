from __future__ import print_function, division

import os
import numpy as np
import pandas as pd

from .config import AIGRAIN_DIR
from .lc import LightCurve

class AigrainLightCurve(LightCurve):
    subdir = 'final'
    def __init__(self, i, ndays=200, sub=10):
        sid = str(int(i)).zfill(4)
        x, y = np.genfromtxt(os.path.join(AIGRAIN_DIR, self.subdir,
                             "lightcurve_{0}.txt".format(sid))).T
        yerr = np.ones(len(y)) * 1e-5
        super(AigrainLightCurve, self).__init__(x - x[0], y - 1, yerr)
        self.restrict_range((0, ndays))
        self.subsample(sub)

    def sigma_clip(self, nsigma=5):
        super(AigrainLightCurve, self).sigma_clip(nsigma)

class AigrainTruths(object):
    filename = os.path.join(AIGRAIN_DIR, 'par', 'final_table.txt')

    def __init__(self):
        self._df = None

    @property
    def df(self):
        if self._df is None:
            self._df = pd.read_table(self.filename, delim_whitespace=True)
        return self._df
