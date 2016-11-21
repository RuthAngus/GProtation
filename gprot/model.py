# This script contains the prior, lhf and logprob functions, plus plotting
# routines.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
import glob
import emcee
import corner
import h5py
import subprocess
import scipy.optimize as spo
import time
import os
import pandas as pd

def lnGauss(x, mu, sigma):
    return -0.5 * ((x - mu)**2/(sigma**2)) + np.log(1./np.sqrt(2*np.pi*sigma**2))

class GPRotModel(object):
    """Parameters are A, l, G, sigma, period
    """
    # log bounds
    _bounds = ((-20., 0.), 
               (-0.69, 20.), 
               (-10., 10.), 
               (-20., 5.), 
               (-0.69, 4.61)) # 0.5 - 100d range

    param_names = ('ln_A', 'ln_l', 'ln_G', 'ln_sigma', 'ln_period')

    def __init__(self, lc, name=None):

        self.lc = lc

        self._name = name

        # Default gaussian for GP param priors
        self.gp_prior_mu = np.array([-13, 6.2, -1.4, -17])
        self.gp_prior_sigma = np.array([5.7, 5.5, 5.5, 5])

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

        loP, hiP = self.bounds[-1]
        samples[:, -1] = np.random.random(N) * (hiP - loP) + loP

        return samples

    def lnprior(self, theta):
        """
        theta = A, l, G, sigma, period
        """
        for i, (lo, hi) in enumerate(self.bounds):
            if theta[i] < lo or theta[i] > hi:
                return -np.inf
        # if not (theta[1] > theta[4] and np.log(0.5) < theta[4]):
        #     return -np.inf

        # p_init = self.plims[1] / 1.5
        # lnpr = lnGauss(theta[4], np.exp(p_init), np.exp(p_init * 0.5))

        lnpr = np.sum(lnGauss(np.array(theta[:4]), 
                              self.gp_prior_mu, self.gp_prior_sigma))

        return lnpr

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

        sigma = np.exp(theta[3])
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
