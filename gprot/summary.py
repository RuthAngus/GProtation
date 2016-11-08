from __future__ import print_function, division

import numpy as np
import pandas as pd

import corner

from .model import GPRotModel

def corner_plot(basename, true_period=None, **kwargs):
    """ Makes corner plot for basename
    """
    df = GPRotModel.get_mnest_samples(basename)
    if len(true_period)==2:
        fig = corner.corner(df, labels=GPRotModel.param_names, **kwargs)
        axes = fig.get_axes()
        axes[-1].axvspan(true_period[0], true_period[1], color='g', alpha=0.3)
    else:
        truths = [None, None, None, None, true_period]
        fig = corner.corner(df, labels=GPRotModel.param_names, 
                        truths=truths, **kwargs)
    return fig