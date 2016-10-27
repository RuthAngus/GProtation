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
As, gammas, ls = [np.zeros(len(truths.N.values[m])) for i in range(3)]
for i, N in enumerate(truths.N.values[m]):
    sN = str(int(N)).zfill(4)
    fname = os.path.join(DIR, "{0}_med_result.txt".format(sN))
    As[i] = np.genfromtxt(fname)[0][0]
    ls[i] = np.genfromtxt(fname)[1][0]
    gammas[i] = np.genfromtxt(fname)[2][0]

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
