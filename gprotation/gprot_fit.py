from __future__ import print_function
import numpy as np
from GProt import mcmc_fit
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py
import math


class fit(object):

    def __init__(self, x, y, yerr, id, sec_size=200, ppd=4):

        self.x = x
        self.y = y
        self.yerr = yerr
        kid = str(int(id)).zfill(9)

        # sigma clip and subsample
        x, y, yerr = sigma_clip(5)
        xb, yb, yerrb = make_gaps(x, y, yerr, ppd)

        # make data into a list of lists, 200 days each
        self.xb, self.yb, self.yerrb = make_lists(xb, yb, yerrb, sec_size)

        # find p_init
        self.acf_period, _, self.pgram_period, _ = calc_p_init(RESULTS_DIR,
                                                               clobber=False)

    def gp_fit(self, RESULTS_DIR, burnin=1000, nwalkers=16, nruns=5,
               full_run=1000):

        # set initial period
        p_init = self.acf_period
        p_max = np.log((self.xb[0][-1] - self.xb[0][0]) / 2.)
        if p_init > np.exp(p_max):
            p_init = 40
        elif p_init < .5:
            p_init = 10
        assert p_init < np.exp(p_max), "p_init > p_max"

        # fast settings
    #     burnin, nwalkers, nruns, full_run = 2, 12, 5, 50
        burnin, nwalkers, nruns, full_run = 500, 12, 3, 100
#         self.xb[0], self.yb[0], self.yerrb[0] = self.xb[0][::100], \
#                 self.yb[0][::100], self.yerrb[0][::100]

        trths = [None, None, None, None, None]
#         mcmc_fit(self.xb[:2], self.yb[:2], self.yerrb[:2], p_init, p_max,
        mcmc_fit(self.xb[0], self.yb[0], self.yerrb[0], p_init, p_max,
                 self.kid, RESULTS_DIR, truths=trths, burnin=burnin,
                 nwalkers=nwalkers, nruns=nruns, full_run=full_run,
                 diff_threshold=.5, n_independent=1000)

    def sigma_clip(self, nsigma):
        med = np.median(self.y)
        std = (sum((med - self.y)**2)/float(len(self.y)))**.5
        m = np.abs(self.y - med) > (nsigma * std)
        return self.x[~m], self.y[~m], self.yerr[~m]

    def make_gaps(self, x, y, yerr, points_per_day):
        nkeep = points_per_day * (x[-1] - x[0])
        m = np.zeros(len(x), dtype=bool)
        l = np.random.choice(np.arange(len(x)), nkeep)
        for i in l:
            m[i] = True
        inds = np.argsort(x[m])
        return x[m][inds], y[m][inds], yerr[m][inds]

    def make_lists(xb, yb, yerrb, l):
        nlists = int(math.ceil((xb[-1] - xb[0]) / l))
        xlist, ylist, yerrlist= [], [], []
        masks = np.arange(nlists + 1) * l
        for i in range(nlists):
            m = (masks[i] < xb) * (xb < masks[i+1])
            xlist.append(xb[m])
            ylist.append(yb[m])
            yerrlist.append(yerrb[m])
        return xlist, ylist, yerrlist

    def calc_p_init(RESULTS_DIR, clobber=False):
        """
        Calculate the ACF and periodogram periods for initialisation.
        """
        fname = os.path.join(RESULTS_DIR, "{0}_acf_pgram_results.txt"
                             .format(self.kid))
        if not clobber and os.path.exists(fname):
            print("Previous ACF pgram result found")
            df = pd.read_csv(fname)
            m = df.N.values == self.kid
            acf_period = df.acf_period.values[m]
            err = df.acf_period_err.values[m]
            pgram_period = df.pgram_period.values[m]
            pgram_period_err = df.pgram_period_err.values[m]
        else:
            print("Calculating ACF")
            acf_period, acf, lags, rvar = sa.simple_acf(self.x, self.y)
            err = .1 * acf_period
            plt.clf()
            plt.plot(lags, acf)
            plt.axvline(acf_period, color="r")
            plt.xlabel("Lags (days)")
            plt.ylabel("ACF")
            plt.savefig(os.path.join(RESULTS_DIR, "{0}_acf".format(id)))
            print("saving figure ", os.path.join(RESULTS_DIR,
                                                 "{0}_acf".format(id)))

            print("Calculating periodogram")
            ps = np.arange(.1, 100, .1)
            model = LombScargle().fit(self.x, self.y, self.yerr)
            pgram = model.periodogram(ps)

            plt.clf()
            plt.plot(ps, pgram)
            plt.savefig(os.path.join(RESULTS_DIR, "{0}_pgram".format(id)))
            print("saving figure ", os.path.join(RESULTS_DIR,
                                                 "{0}_pgram".format(id)))

            peaks = np.array([i for i in range(1, len(ps)-1) if pgram[i-1] <
                              pgram[i] and pgram[i+1] < pgram[i]])
            pgram_period = ps[pgram == max(pgram[peaks])][0]
            print("pgram period = ", pgram_period, "days")
            pgram_period_err = pgram_period * .1

            df = pd.DataFrame({"N": [id], "acf_period": [acf_period],
                               "acf_period_err": [err],
                               "pgram_period": [pgram_period],
                               "pgram_period_err": [pgram_period_err]})
            df.to_csv(fname)
        return acf_period, err, pgram_period, pgram_period_err
