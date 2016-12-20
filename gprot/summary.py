from __future__ import print_function, division

import os, glob, re
import numpy as np
import pandas as pd
from collections import OrderedDict

import corner

from .model import GPRotModel, GPRotModel2

def corner_plot(df, mod, true_period=None, pct=0.99, **kwargs):
    """ Makes corner plot for basename
    """
    ndim = len(mod.param_names)
    labels = mod.param_names

    rng = [pct]*ndim
    truths = [None]*ndim
    if true_period is not None and len(true_period)==2:
        P1, P2 = true_period
        rng[-1] = (min(df['ln_period'].quantile(1-pct), P1)-0.01,
                    max(df['ln_period'].quantile(pct), P2)+0.01)
        fig = corner.corner(df, labels=labels, range=rng, **kwargs)
        axes = fig.get_axes()
        axes[-1].axvspan(P1, P2, color='g', alpha=0.3)
    elif true_period is not None:
        rng[-1] = (min(df['ln_period'].quantile(1-pct), true_period)-0.01,
                    max(df['ln_period'].quantile(pct), true_period)+0.01)
        truths[-1] = true_period
        fig = corner.corner(df, labels=labels, range=rng,
                        truths=truths, **kwargs)
    else:
        fig = corner.corner(df, labels=labels, range=rng, **kwargs)

    return fig

def summarize_fits(directory, quantiles=[0.05,0.16,0.5,0.84,0.95],
                   truths=None):
    d = {}
    names = [os.path.basename(f)[:-3] for f in glob.glob(os.path.join(directory,'*.h5'))]
    for name in names:
        try:
            name = int(name)
        except ValueError:
            pass
        d[name] = OrderedDict()
        sample_file = os.path.join(directory, '{}.h5'.format(name))
        samples = pd.read_hdf(sample_file, 'samples')
        quants = samples.quantile(quantiles)
        for c in samples.columns:
            for q in quantiles:
                key = '{}_{:02.0f}'.format(c, q*100)
                d[name][key] = quants.ix[q, c]
    df = pd.DataFrame.from_dict(d, orient='index')    
    df.sort_index()

    if truths=='aigrain':
        from .aigrain import AigrainTruths
        true_df = AigrainTruths().df
        df['aigrain_p_min'] = np.log(true_df.ix[df.index, 'P_MIN'])
        df['aigrain_p_max'] = np.log(true_df.ix[df.index, 'P_MAX'])
        df['aigrain_p_mean'] = (df['aigrain_p_min'] + df['aigrain_p_max'])/2.

    return df

def digest_acf(df):
    """Picks best ACF prot, height, quality, tau
    """
    

