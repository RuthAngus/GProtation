import numpy as np
import acor

from .peakdetect import peakdetect

def acf(x, y, maxlag=100):
    """Assumes regular sampling, zero-filled
    """
    cadence = np.median(np.diff(x))
    maxlag_cad = maxlag/cadence
    ac = acor.function(y, maxlag_cad)
    lags = np.arange(len(ac))*cadence
    return lags, ac

def acf_prot(x, y, maxlag=100, delta=0.02, lookahead=30):
    """Returns best guess of prot from ACF

    Returns highest trough-to-peak-height peak of first two peaks.
    """
    lags, ac = acf(x, y, maxlag=maxlag)

    maxes, mins = peakdetect(ac, lags)

    maxheight = 0
    pbest = None
    for (xhi, yhi), (xlo, ylo) in zip(maxes, mins):
        height = yhi - ylo
        if height > maxheight:
            pbest = xhi

    return pbest, maxheight