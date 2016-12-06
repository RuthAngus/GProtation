import numpy as np
import acor

def acf(x, y, maxlag=100):
    """Assumes regular sampling, zero-filled
    """
    cadence = np.median(np.diff(x))
    maxlag_cad = maxlag/cadence
    ac = acor.function(y, maxlag_cad)
    lags = np.arange(len(ac))*cadence
    return lags, ac

