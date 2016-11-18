import os, sys
import pandas as pd
import numpy as np

def write_samples(mod, samples, resultsdir='results'):
    pass

def fit_mnest(mod, basename=None, test=False, 
                verbose=False, resultsdir='results', **kwargs):
    import pymultinest

    if basename is None:
        basename = os.path.join('chains', mod.name)

    if test:
        print('Will run multinest on star {}..., basename={}'.format(i, basename))
    else:
        _ = pymultinest.run(mod.mnest_loglike, mod.mnest_prior, 5, 
                            verbose=verbose, outputfiles_basename=basename, 
                            **kwargs)

        if not os.path.exists(resultsdir):
            os.makedirs(resultsdir)

        df = GPRotModel.get_mnest_samples(basename)
        samplefile = os.path.join(resultsdir, '{}.h5'.format(i))
        df.to_hdf(samplefile, 'samples')
        mod.lc.df.to_hdf(samplefile, 'lc')
        prior_df = pd.DataFrame({'mu':mod.gp_prior_mu,
                                 'sigma':mod.gp_prior_sigma})
        prior_df.to_hdf(samplefile, 'gp_prior')

        print('Samples, light curve, and prior saved to {}.'.format(samplefile))
        figfile = os.path.join(resultsdir, '{}.png'.format(i))
        fig = lc.corner_plot()
        fig.savefig(figfile)
        print('Corner plot saved to {}.'.format(figfile))
