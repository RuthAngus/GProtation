import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)

prior = True

if prior:
    RESULTS_DIR = "results_24_01"
else:
    RESULTS_DIR = "results_noprior"
truths = pd.read_csv("final_table.txt", delimiter=" ")

# remove differential rotators and take just the first 100
m = truths.DELTA_OMEGA.values == 0
truths = truths.iloc[m]
# ml = np.arange(100)
# truths = truths.iloc[ml]

recovered = np.zeros(len(truths.N.values))
errp, errm = [np.zeros(len(truths.N.values)) for i in range(2)]
lnerrp, lnerrm = [np.zeros(len(truths.N.values)) for i in range(2)]
for i, id in enumerate(truths.N.values):
    fn = os.path.join(RESULTS_DIR, "{}.h5".format(id))
    if os.path.exists(fn):
        # store = pd.HDFStore(fn)
        # print(store)
        df = pd.read_hdf(fn, key="samples")
        phist, bins = np.histogram(df.ln_period.values, 100)
        ln_p = bins[phist == max(phist)][0]
        # ln_p = np.median(df.ln_period.values)
        recovered[i] = np.exp(ln_p)
        lnerrp[i] = np.percentile(df.ln_period.values, 84) - ln_p
        lnerrm[i] = ln_p - np.percentile(df.ln_period.values, 16)
        errp[i] = np.exp(lnerrp[i]/ln_p)
        errm[i] = np.exp(lnerrm[i]/ln_p)

x = .5 * (truths.P_MIN.values + truths.P_MAX.values)
amp = truths.AMP.values
l = recovered > 0

plt.clf()
xs = np.log(np.linspace(0, 55, 100))
plt.plot(xs, xs, "-", color=".7", zorder=0)
plt.plot(xs, xs + 2./3, "--", color=".7", zorder=0)
plt.plot(xs, xs - 2./3, "--", color=".7", zorder=0)

plt.errorbar(np.log(x[l]), np.log(recovered[l]), yerr=[lnerrp[l], lnerrm[l]],
             fmt="k.", zorder=1, capsize=0, ecolor=".7", alpha=.5, ms=.1)
plt.scatter(np.log(x[l]), np.log(recovered[l]), c=np.log(amp[l]),
            edgecolor=".5", cmap="GnBu_r", vmin=min(np.log(amp[l])),
            vmax=max(np.log(amp[l])), s=20, zorder=2, lw=.2)
plt.plot(xs, xs + 2./3, "--", color=".7", zorder=0)
plt.plot(xs, xs + 2./3, "--", color=".7", zorder=0)
plt.xlim(0, 4)
plt.ylim(0, 6)

cbar = plt.colorbar()
cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
plt.xlabel("$\ln(\mathrm{Injected~Period})$")
plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
if prior:
    plt.savefig("comparison_emcee3_24_0")
else:
    plt.savefig("comparison_noprior")
    plt.savefig("../documents/figures/comparison_noprior.pdf")

plt.clf()
resids = np.log(x[l]) - np.log(recovered[l])
plt.hist(resids, 80, histtype="stepfilled", color="w")
plt.xlabel("$\ln(\mathrm{True~Period}) - \ln(\mathrm{GP~Period})$")
plt.axvline(np.percentile(resids, 16), color=".5", ls="--")
plt.axvline(np.percentile(resids, 84), color=".5", ls="--")

median_err = .5*(np.median(lnerrp[l]) + np.median(lnerrm[l]))
median_err = np.median(.5*(lnerrp[l] + lnerrm[l]))
plt.errorbar(-1, 100, xerr=median_err, fmt="k.", ms=.1)
if prior:
    plt.savefig("gp_hist.pdf")
else:
    plt.savefig("gp_hist_noprior.pdf")


print("percentiles:")
print(np.percentile(resids, 16), np.percentile(resids, 84))
print("diff = ",
      np.abs(np.percentile(resids, 16) - np.percentile(resids, 84)))
print("median_err = ", median_err)

print("MAD = ", (np.median((x[l] - recovered[l])**2)**.5))
print("MAD = ", (np.median((np.log(x[l]) - np.log(recovered[l]))**2)**.5))

errs = .5*(lnerrp[l] + lnerrm[l])
plt.clf()
plt.hist(errs, 100)
if prior:
    plt.savefig("err_hist")
else:
    plt.savefig("err_hist_noprior")

plt.clf()
nsigma_diff = np.abs(resids - errs)/errs
plt.hist(nsigma_diff, 100, histtype="stepfilled", color="w")
plt.axvline(np.percentile(nsigma_diff, 66), color="r", ls="--")
print(np.percentile(nsigma_diff, 66))
print(max(nsigma_diff))
if prior:
    plt.savefig("err_resid_ratio_hist")
else:
    plt.savefig("err_resid_ratio_hist_noprior")

"""
3/4 of uncertainties are under-estimated.
1/2 are within 2 sigma.
66% are within 3 sigma.
Largest outlier is 114 sigma off.
"""

print((np.median(np.abs(recovered[l] - x[l]))))