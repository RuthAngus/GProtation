import numpy as np
import matplotlib.pyplot as plt
from plotstuff import params, colours
reb = params()
cols = colours()

plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 20,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)

data = np.genfromtxt("../par/final_table.txt", skip_header=1).T
m = data[13] == 0  # just the stars without diffrot
ids = data[0][m]
periods = data[-3][m]
star = 1
print("period = ", periods[star])
id = str(int(ids[star])).zfill(4)
x, y = np.genfromtxt("../noise_free/lightcurve_{0}.txt".format(id)).T
y -= np.mean(y)

plt.clf()
plt.plot(x, y, "k")
plt.xlabel("$\mathrm{Time~(days)}$")
plt.ylabel("$\mathrm{Normalised~flux}$")
plt.subplots_adjust(left=.2, bottom=.12)
plt.xlim(min(x), max(x))
plt.savefig("thesis_plot")
