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

DIR = "../code/simulations/kepler_diffrot_full/par/"
truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
m = truths.DELTA_OMEGA.values == 0

DIR = "results"
As, gammas, ls, ss, ps = [np.zeros(len(truths.N.values[m])) for i in range(5)]
for i, N in enumerate(truths.N.values[m]):
    sN = str(int(N)).zfill(4)
    fname = os.path.join(DIR, "{0}_med_result.txt".format(sN))
    data = np.genfromtxt(fname)
    As[i] = data[0][0]
    ls[i] = data[1][0]
    gammas[i] = data[2][0]
    ss[i] = data[3][0]
    ps[i] = data[4][0]

print("median ln gamma = ", np.median(gammas), "std = ", np.std(gammas))
print("median gamma = ", np.exp(np.median(gammas)), "std = ",
      np.exp(np.std(gammas)))
plt.clf()
plt.hist(gammas, 50, histtype="stepfilled", color="w")
plt.xlabel("$\ln(\Gamma)$")
plt.savefig("gamma_hist")

print("median ln A = ", np.median(As), "std = ", np.std(As))
print("median A = ", np.exp(np.median(As)), "std = ",
      np.exp(np.std(As)))
plt.clf()
plt.hist(As, 30, histtype="stepfilled", color="w")
plt.xlabel("$\ln(A)$")
plt.savefig("A_hist")

print("median ln l = ", np.median(ls), "std = ", np.std(ls))
print("median l = ", np.exp(np.median(ls)), "std = ",
      np.exp(np.std(ls)))
plt.clf()
plt.hist(ls, 30, histtype="stepfilled", color="w")
plt.xlabel("$\ln(l)$")
plt.savefig("l_hist")

print("median ln sigma = ", np.median(ss), "std = ", np.std(ss))
print("median sigma = ", np.exp(np.median(ss)), "std = ",
      np.exp(np.std(ss)))
plt.clf()
plt.hist(ss, 30, histtype="stepfilled", color="w")
plt.xlabel("$\ln(\sigma)$")
plt.savefig("sigma_hist")

print("median ln period = ", np.median(ps), "std = ", np.std(ps))
print("median period = ", np.exp(np.median(ps)), "std = ",
      np.exp(np.std(ps)))
plt.clf()
plt.hist(ps, 30, histtype="stepfilled", color="w")
plt.xlabel("$\ln(Period)$")
plt.savefig("period_hist")
