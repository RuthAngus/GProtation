import os, sys
import pandas as pd
import numpy as np

import emcee3

class Emcee3Model(emcee3.Model):
    def __init__(self, mod, *args, **kwargs):
        self.mod = mod
        super(Emcee3Model, self).__init__(*args, **kwargs)

    def compute_log_prior(self, state):
        state.log_prior = self.mod.lnprior(state.coords)
        return state

    def compute_log_likelihood(self, state):
        state.log_likelihood = self.mod.lnlike(state.coords)
        return state

def write_samples(mod, df, resultsdir='results'):
    """df is dataframe of samples, mod is model
    """

    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)
    samplefile = os.path.join(resultsdir, '{}.h5'.format(mod.name))
    df.to_hdf(samplefile, 'samples')
    mod.lc.df.to_hdf(samplefile, 'lc')
    prior_df = pd.DataFrame({'mu':mod.gp_prior_mu,
                             'sigma':mod.gp_prior_sigma})
    prior_df.to_hdf(samplefile, 'gp_prior')

    print('Samples, light curve, and prior saved to {}.'.format(samplefile))
    figfile = os.path.join(resultsdir, '{}.png'.format(mod.name))
    fig = lc.corner_plot()
    fig.savefig(figfile)
    print('Corner plot saved to {}.'.format(figfile))

def fit_emcee3(mod, nwalkers=500, verbose=False, nsamples=5000, targetn=6,
                iter_chunksize=10, processes=None):
    """fit model using Emcee3 

    modeled after https://github.com/dfm/gaia-kepler/blob/master/fit.py

    """
    #initialize
    walker = Emcee3Model(mod)
    ndim = 5

    coords_init = mod.sample_from_prior(nwalkers)
    sampler = emcee3.Sampler(emcee3.moves.KDEMove())

    if processes is not None:
        from multiprocessing import Pool
        pool = Pool(processes=processes)
    else:
        from emcee3.pool import DefaultPool
        pool = DefaultPool()

    ensemble = emcee3.Ensemble(walker, coords_init, pool=pool)

    chunksize = iter_chunksize
    for iteration in range(100):
        if verbose:
            print("Iteration {0}...".format(iteration + 1))
        sampler.run(ensemble, chunksize, progress=verbose)
        mu = np.mean(sampler.get_coords(), axis=1)
        try:
            tau = emcee3.autocorr.integrated_time(mu, c=1)
        except emcee3.autocorr.AutocorrError:
            continue
        tau_max = tau.max()
        neff = ((iteration+1) * chunksize / tau_max - 2.0)
        if verbose:
            print("Maximum autocorrelation time: {0}".format(tau_max))
            print("N_eff: {0}\n".format(neff * nwalkers))
        if neff > targetn:
            break

    burnin = int(2*tau_max)
    ntot = nsamples
    if verbose:
        print("Discarding {0} samples for burn-in".format(burnin))
        print("Randomly choosing {0} samples".format(ntot))
    samples = sampler.get_coords(flat=True, discard=burnin)
    total_samples = len(samples)
    inds = np.random.choice(np.arange(len(samples)), size=ntot, replace=False)
    samples = samples[inds]

    df = pd.DataFrame(samples, columns=mod.param_names)
    write_samples(mod, df)
    return df

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
        
        write_samples(mod, df)
        return df