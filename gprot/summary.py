from __future__ import print_function, division

import numpy as np
import pandas as pd

import corner

from .model import GPRotModel

def corner_plot(df, true_period=None, **kwargs):
    """ Makes corner plot for basename
    """
    if len(true_period)==2:
        P1, P2 = true_period
        rng = [0.999, 0.999, 0.999, 0.999, (min(df['ln_period'].min(), P1)-0.01,
                                            max(df['ln_period'].max(), P2)+0.01)]
        fig = corner.corner(df, labels=GPRotModel.param_names, range=rng, **kwargs)
        axes = fig.get_axes()
        axes[-1].axvspan(P1, P2, color='g', alpha=0.3)
    else:
        rng = [0.999, 0.999, 0.999, 0.999, (min(df['ln_period'].min(), true_period)-0.01,
                                            max(df['ln_period'].max(), true_period)+0.01)]
        truths = [None, None, None, None, true_period]
        fig = corner.corner(df, labels=GPRotModel.param_names, 
                        truths=truths, **kwargs)
    return fig