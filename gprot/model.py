# This script contains the prior, lhf and logprob functions, plus plotting
# routines.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel, CosineKernel
import glob
import emcee
import corner
import h5py
import subprocess
import scipy.optimize as spo
import time
import os
import pandas as pd
from scipy.misc import logsumexp

def lnGauss(x, mu, sigma):
    return -0.5 * ((x - mu)**2/(sigma**2)) + np.log(1./np.sqrt(2*np.pi*sigma**2))

def lnGauss_mixture(x, mix):
    w = mix[:, 0]
    mu = mix[:, 1]
    sigma = mix[:, 2]
    return logsumexp(lnGauss(x, mu, sigma), b=w)
    # return np.log(np.sum([w*np.exp(lnGauss(x, mu, sig)) for w, mu, sig in mix]))

class GPRotModel(object):
    """Parameters are A, l, G, sigma, period
    """
    # log bounds
    _bounds = ((-20., 0.), 
               (2, 20.), 
               (-10., 3.), 
               (-20., 0.), 
               (-0.69, 4.61)) # 0.5 - 100d range

    param_names = ('ln_A', 'ln_l', 'ln_G', 'ln_sigma', 'ln_period')

    _default_gp_prior_mu = (-13, 7.2, -2.3, -17)
    _default_gp_prior_sigma = (5.7, 1.2, 1.4, 5)

    _acf_pmax = (1,3,5,10,30,50,100)
    _acf_prior_width = 0.2

    def __init__(self, lc, name=None, pmin=None, pmax=None,
                 acf_prior=False):

        self.lc = lc

        self._name = name

        if pmin is None:
            pmin = -np.inf
        if pmax is None:
            pmax = np.inf
        self.pmin = pmin
        self.pmax = pmax

        self.acf_prior = acf_prior

        # Default gaussian for GP param priors
        self.gp_prior_mu = np.array(self._default_gp_prior_mu)
        self.gp_prior_sigma = np.array(self._default_gp_prior_sigma)

    @property
    def ndim(self):
        return len(self.param_names)

    @property
    def x(self):
        return self.lc.x

    @property
    def y(self):
        return self.lc.y

    @property
    def yerr(self):
        return self.lc.yerr

    @property
    def name(self):
        if self._name is None:
            return self.lc.name
        else:
            return self._name

    @property
    def bounds(self):
        return self._bounds

    def sample_from_prior(self, N, seed=None):
        """
        Returns N x ndim array of prior samples
        (within bounds)
        """
        samples = np.empty((N, self.ndim))

        np.random.seed(seed)
        for i in range(self.ndim - 1):
            vals = np.inf*np.ones(N)
            m = ~np.isfinite(vals)
            nbad = m.sum()
            while nbad > 0:
                rn = np.random.randn(nbad)
                vals[m] = rn*self.gp_prior_sigma[i] + self.gp_prior_mu[i]
                m = (vals < self.bounds[i][0]) | (vals > self.bounds[i][1])
                nbad = m.sum()

            samples[:, i] = vals

        samples[:, -1] = self.sample_period_prior(N)

        return samples

    def lnprior(self, theta):
        """
        theta = A, l, G, sigma, period
        """
        for i, (lo, hi) in enumerate(self.bounds):
            if theta[i] < lo or theta[i] > hi:
                return -np.inf
        if theta[-1] < self.pmin or theta[-1] > self.pmax:
            return -np.inf

        # Don't let SE correlation length be shorter than P.
        # if theta[1] < theta[-1]:
        #     return -np.inf

        # if not (theta[1] > theta[4] and np.log(0.5) < theta[4]):
        #     return -np.inf

        # p_init = self.plims[1] / 1.5
        # lnpr = lnGauss(theta[4], np.exp(p_init), np.exp(p_init * 0.5))

        lnpr = np.sum(lnGauss(np.array(theta[:-1]), 
                              self.gp_prior_mu, self.gp_prior_sigma))

        lnpr += self.lnprior_period(theta[-1])

        return lnpr

    def lnprior_period(self, p):
        if not self.acf_prior:
            return 0
        else:
            return lnGauss_mixture(p, self.period_mixture)

    def sample_period_prior(self, N):
        loP, hiP = self.bounds[-1]
        if self.pmin > loP:
            loP = self.pmin
        if self.pmax < hiP:
            hiP = self.pmax
        if not self.acf_prior:
            vals = np.random.random(N) * (hiP - loP) + loP
        else:
            vals = np.empty(N)
            u = np.random.random(N)
            lo = 0
            for w, mu, sig in self.period_mixture:
                hi = lo + w
                ok = (u > lo) & (u < hi)
                n = ok.sum()

                # Make sure periods are in range
                newvals = np.inf*np.ones(n)
                m = (newvals < loP) | (newvals > hiP)
                nbad = m.sum()
                while nbad > 0:
                    rn = np.random.randn(nbad)
                    newvals[m] = rn*sig + mu
                    m = (newvals < loP) | (newvals > hiP)
                    nbad = m.sum()

                vals[ok] = newvals
                lo += w

        return vals

    @property
    def acf_results(self):
        if not hasattr(self, '_acf_results'):
            self._calc_acf()
        return self._acf_results

    @property
    def acf_pmax(self):
        return self._acf_pmax

    @property
    def acf_prior_width(self):
        return self._acf_prior_width

    def _calc_acf(self):
        self._acf_results = [self.lc.acf_prot(pmax=p) for p in self.acf_pmax]

    def _lnp_in_bounds(self, lnp):
        return lnp > self.bounds[-1][0] and lnp < self.bounds[-1][1]

    @property
    def period_mixture(self):
        """Gaussian mixture describing period prior

        list of (weight, mu, sigma)
        """
        if not hasattr(self, '_period_mixture'):
            self._period_mixture = []
            ln2 = np.log(2)
            wtot = 0
            for p, _, _, w in self.acf_results:
                if np.isfinite(w):
                    wtot += w
            for p, _, _, w in self.acf_results:
                if not np.isfinite(w):
                    continue
                wi = w/wtot
                # add 90% at this period, and 5% at twice and half
                lnp = np.log(p)
                lnplo = lnp - ln2
                lnphi = lnp + ln2
                if self._lnp_in_bounds(lnp):
                    self._period_mixture.append((0.9*wi, lnp, self.acf_prior_width))
                if self._lnp_in_bounds(lnplo):
                    self._period_mixture.append((0.05*wi, lnplo, self.acf_prior_width))
                if self._lnp_in_bounds(lnphi):
                    self._period_mixture.append((0.05*wi, lnphi, self.acf_prior_width))
            self._period_mixture = np.array(self._period_mixture)

        return self._period_mixture

    def gp_kernel(self, theta):
        A = np.exp(theta[0])
        l = np.exp(theta[1])
        G = np.exp(theta[2])
        sigma = np.exp(theta[3])
        P = np.exp(theta[4])
        return A * ExpSquaredKernel(l) * ExpSine2Kernel(G, P) + WhiteKernel(sigma)        

    def gp(self, theta, x=None, yerr=None):
        if x is None:
            x = self.x
        if yerr is None:
            yerr = self.yerr

        k = self.gp_kernel(theta)
        gp = george.GP(k, solver=george.HODLRSolver)

        sigma = np.exp(theta[-2])
        gp.compute(x, np.sqrt(sigma + yerr**2)) # is this correct?
        return gp

    def lnlike_function(self, theta, x, y, yerr):
        try:
            gp = self.gp(theta, x=x, yerr=yerr)
        except (ValueError, np.linalg.LinAlgError):
            return -np.inf
        lnl = gp.lnlikelihood(y, quiet=True)

        return lnl if np.isfinite(lnl) else -np.inf

    def lnlike(self, theta):
        if self.lc.x_list is None:
            return self.lnlike_function(theta, self.x, self.y, self.yerr)
        else:
            return np.sum([self.lnlike_function(theta, x=x, y=y, yerr=yerr)
                            for (x, y, yerr) in zip(self.lc.x_list,
                                                    self.lc.y_list,
                                                    self.lc.yerr_list)])

    def lnpost(self, theta):
        lnprob = self.lnlike(theta) + self.lnprior(theta)
        return lnprob

    def polychord_prior(self, cube):
        """Takes unit cube, returns true parameters
        """
        return [lo + (hi - lo)*cube[i] for i, (lo, hi) in enumerate(self.bounds)]

    def polychord_lnpost(self, theta):
        phi = [0.0] * 0
        return self.lnpost(theta), phi

    def mnest_prior(self, cube, ndim, nparams):
        for i in range(5):
            lo, hi = self.bounds[i]
            cube[i] = (hi - lo)*cube[i] + lo

    def mnest_loglike(self, cube, ndim, nparams):
        """loglikelihood function for multinest
        """
        return self.lnpost(cube)

    @classmethod
    def get_mnest_samples(cls, basename):
        """Returns pandas DataFrame of samples created by MultiNest
        """
        post_file = '{}post_equal_weights.dat'.format(basename)
        data = np.loadtxt(post_file)
        return pd.DataFrame(data[:,:-1], columns=cls.param_names)

class GPRotModel2(GPRotModel):
    """ Playing with model a bit...
    """
    _bounds = ((-20., 0.), 
               (-0.69, 20.), 
               (-20., 5.), 
               (-0.69, 4.61)) # 0.5 - 100d range

    param_names = ('ln_A', 'ln_l', 'ln_sigma', 'ln_period')

    _default_gp_prior_mu = (-13, 6.2, -17)
    _default_gp_prior_sigma = (7.7, 5.5, 5)

    def gp_kernel(self, theta):
        A = np.exp(theta[0])
        l = np.exp(theta[1])
        sigma = np.exp(theta[2])
        P = np.exp(theta[3])
        return A * ExpSquaredKernel(l) * CosineKernel(P) + WhiteKernel(sigma)        
