import numpy as np
import matplotlib.pyplot as plt
import glob

plotpar = {'axes.labelsize': 18,
           'font.size': 18,
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)

orange = "#FF9933"
blue = "#0066CC"
pink = "#FF33CC"
green = "#009933"
cols = [blue, pink, orange, green]

fnames = glob.glob("simulations/????.txt")

plt.clf()
f = [fnames[0], fnames[20], fnames[40]]
for i, fname in enumerate(f):
    x, y = np.genfromtxt(fname).T
    plt.plot(x[:7000], y[:7000], color=cols[i], lw=2)
plt.xlim(min(x), max(x[:7000]))
plt.xlabel("$\mathrm{Time~(days)}$")
plt.ylabel("$\mathrm{Flux}$")
plt.savefig("penn_state.pdf")
