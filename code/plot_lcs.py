# Script for generating thesis figures.
import numpy as np
import matplotlib.pyplot as plt
from plotstuff import params, colours
cols = colours()
reb = params()
from simple_acf import dan_acf

plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 20,
           'xtick.labelsize': 20,
           'ytick.labelsize': 20,
           'text.usetex': True}
plt.rcParams.update(plotpar)

orange = '#FF9933'
lightblue = '#66CCCC'
blue = '#0066CC'
pink = '#FF33CC'
turquoise = '#3399FF'
lightgreen = '#99CC99'
green = '#009933'
maroon = '#CC0066'
purple = '#9933FF'
red = '#CC0000'
lilac = '#CC99FF'

ids, ps, amps = np.genfromtxt("simulations/noise-free/true_periods_amps.txt").T
id = "0006"
x, y = np.genfromtxt("simulations/noise-free/{}.txt".format(id)).T
plt.clf()
plt.plot(x, y, "k-", label = "$\mathrm{Period} = %.1f~\mathrm{days}$" % ps[int(id)])
plt.xlabel("$\mathrm{Time~(days)}$")
plt.ylabel("$\mathrm{Normalised~flux}$")
plt.xlim(0, 200)
plt.legend()
plt.subplots_adjust(bottom=.12)
plt.savefig("../figures/noise-free_lc.pdf")

# ACF figure
gap_days = .02043365
acf = dan_acf(y)
lags = np.arange(len(acf)) * gap_days
plt.clf()
plt.plot(lags, acf, "k")
plt.xlim(0, 100)
plt.xlabel("$\mathrm{Time~(days)}$")
plt.ylabel("$\mathrm{Correlation}$")
_, true_periods, _ = np.genfromtxt("simulations/true_periods.txt").T
# plt.title("$\mathrm{P}_{\mathrm{rot}} = {0}\mathrm{days}$".format(true_periods[0]))
plt.savefig("../figures/noise-free_acf.pdf")

x2, y2, yerr = np.genfromtxt("simulations/kepler_injections/{}.txt".format(id)).T
plt.clf()
plt.plot(x2, y2, "k.", label = "$\mathrm{Period} = %.1f~\mathrm{days}$"
         % ps[0])
plt.plot(x, y, color=lightblue, label =
         "$\mathrm{Period} = %.1f~\mathrm{days}$" % ps[0])
plt.xlabel("$\mathrm{Time~(days)}$")
plt.ylabel("$\mathrm{Normalised~flux}$")
plt.xlim(0, 200)
plt.legend()
plt.subplots_adjust(bottom=.12)
plt.savefig("../figures/noisy_lc.pdf")
