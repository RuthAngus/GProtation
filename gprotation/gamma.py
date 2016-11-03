# plot histograms of the "other" parameters.
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def make_gamma_plots(ids, DIR):

    As, gammas, ls, ss, ps = [np.zeros(len(ids)) for i in range(5)]
    for i, N in enumerate(ids):
        sN = str(int(N)).zfill(4)
        fname = os.path.join(DIR, "{0}_mcmc_results.csv".format(sN))
        if os.path.exists(fname):
            data = pd.read_csv(fname)
            As[i] = data.A_max
            ls[i] = data.l_max
            gammas[i] = data.gamma_max
            ss[i] = data.period_max
            ps[i] = data.sigma_max

    print("median ln gamma = ", np.median(gammas), "std = ", np.std(gammas))
    print("median gamma = ", np.exp(np.median(gammas)), "std = ",
          np.exp(np.std(gammas)))
    plt.clf()
    plt.hist(gammas, 50, histtype="stepfilled", color="w")
    plt.xlabel("$\ln(\Gamma)$")
    plt.savefig(os.path.join(DIR, "gamma_hist"))

    print("median ln A = ", np.median(As), "std = ", np.std(As))
    print("median A = ", np.exp(np.median(As)), "std = ",
          np.exp(np.std(As)))
    plt.clf()
    plt.hist(As, 30, histtype="stepfilled", color="w")
    plt.xlabel("$\ln(A)$")
    plt.savefig(os.path.join(DIR, "A_hist"))

    print("median ln l = ", np.median(ls), "std = ", np.std(ls))
    print("median l = ", np.exp(np.median(ls)), "std = ",
          np.exp(np.std(ls)))
    plt.clf()
    plt.hist(ls, 30, histtype="stepfilled", color="w")
    plt.xlabel("$\ln(l)$")
    plt.savefig(os.path.join(DIR, "l_hist"))

    print("median ln sigma = ", np.median(ss), "std = ", np.std(ss))
    print("median sigma = ", np.exp(np.median(ss)), "std = ",
          np.exp(np.std(ss)))
    plt.clf()
    plt.hist(ss, 30, histtype="stepfilled", color="w")
    plt.xlabel("$\ln(\sigma)$")
    plt.savefig(os.path.join(DIR, "sigma_hist"))

    print("median ln period = ", np.median(ps), "std = ", np.std(ps))
    print("median period = ", np.exp(np.median(ps)), "std = ",
          np.exp(np.std(ps)))
    plt.clf()
    plt.hist(ps, 30, histtype="stepfilled", color="w")
    plt.xlabel("$\ln(Period)$")
    plt.savefig(os.path.join(DIR, "period_hist"))

    plt.clf()
    plt.plot(ps, gammas, "k.")
    plt.xlabel("ln(period)")
    plt.ylabel("ln(gamma)")
    plt.xlim(np.log(.5), np.log(150))
    plt.ylim(-10, 10)
    plt.savefig(os.path.join(DIR, "period_vs_gamma"))

    plt.clf()
    plt.plot(ps, ls, "k.")
    plt.ylabel("ln(l)")
    plt.xlabel("ln(period)")
    plt.xlim(np.log(.5), np.log(150))
    plt.ylim(0, 20)
    plt.savefig(os.path.join(DIR, "period_vs_l"))

    plt.clf()
    plt.plot(ls, gammas, "k.")
    plt.xlabel("ln(l)")
    plt.ylabel("ln(gamma)")
    plt.ylim(0, 20)
    plt.ylim(-10, 10)
    plt.savefig(os.path.join(DIR, "l_vs_gamma"))

if __name__ == "__main__":

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0
    make_gamma_plots(truths.N.values[m], "results")
    make_gamma_plots(truths.N.values[m], "results_Gprior")

    make_gamma_plots(np.arange(333), "results_periodic_gp")
    make_gamma_plots(np.arange(333), "results_aperiodic_gp")
