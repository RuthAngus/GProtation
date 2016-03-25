import numpy as np
import matplotlib.pyplot as plt
from plotstuff import params, colours
reb = params()


def expcos(x, p, a, l):
    return a*np.cos(2*np.pi*(1./p)*x) + np.exp(-.5*(x/l)**2)


# generate exponential sine waves
N = 1000
ps = 10**np.random.uniform(0, 2, N)
ls = 10**np.random.uniform(1, 2, N)
aa = 10**np.random.uniform(-1, 1, N)
aa = np.ones_like(ps)*.1
# ls = np.ones_like(ps)*100

x = np.linspace(0, 100, 1000)
periods = []
for i, p in enumerate(ps):
    y = expcos(x, p, aa[i], ls[i])
    peaks = np.array([i for i in range(1, len(x)-1) if y[i-1] < y[i]
                     and y[i+1] < y[i]])
    period = 0
    if len(peaks):
        period = x[y == max(y[peaks])]
    periods.append(period)
    if i == 0:
        plt.clf()
        plt.plot(x, y)
        plt.axvline(period, color="r")
        plt.savefig("test")

periods = np.array(periods)
m = periods > 0
plt.clf()
xs = np.linspace(0, 100, 100)
plt.plot(xs, xs, "--", color=".7")
plt.plot(xs, .5*xs, "--", color=".7")
plt.plot(xs, 2*xs, "--", color=".7")
plt.plot(ps[m], periods[m], "k.")
plt.ylim(0, 120)
plt.xlabel("$\mathrm{True~period~(days)}$")
plt.ylabel("$\mathrm{Measured~period~(days)}$")
plt.savefig("exp_sine_test.pdf")
