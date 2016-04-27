from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import h5py
import triangle
import george
from george.kernels import ExpSquaredKernel, ExpSine2Kernel

plotpar = {'axes.labelsize': 24,
           'xtick.labelsize': 16,
           'ytick.labelsize': 16,
           'text.usetex': True}
plt.rcParams.update(plotpar)

id = 2

with h5py.File("sine/000{0}_samples.h5".format(id), "r") as f:
    samples = f["samples"][...]
nwalkers, nsteps, ndims = np.shape(samples)
flat = np.reshape(samples, (nwalkers*nsteps, ndims))
fig_labels = ["$\log(A)$", "$\log(l)$", "$\log(\Gamma)$", "$\log(\sigma)$",
              "$P_{\mathrm{rot}}$"]
flat[:, -1] = np.exp(flat[:, -1])
# fig = triangle.corner(flat, labels=fig_labels)
# plt.savefig("demo_triangle.pdf")

mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                  zip(*np.percentile(flat, [16, 50, 84], axis=0)))
theta = np.exp(np.array([mcmc_result[i][0] for i in range(ndims)]))
print("plotting prediction")
theta = np.exp(np.array(mcmc_result))
k = theta[0] * ExpSquaredKernel(theta[1]) \
        * ExpSine2Kernel(theta[2], theta[4])
gp = george.GP(k, solver=george.HODLRSolver)
gp.compute(xb, yerrb)
xs = np.linspace(xb[0], xb[-1], 1000)
mu, cov = gp.predict(yb, xs)
plt.clf()
plt.errorbar(x-x[0], y, yerr=yerr, **reb)
#         plt.errorbar(xb, yb, yerr=yerrb, fmt="r.")
plt.xlabel("$\mathrm{Time~(days)}$")
plt.ylabel("$\mathrm{Normalised~Flux}$")
plt.plot(xs, mu, color=cols.lightblue)
plt.xlim(min(xb), max(xb))
#         plt.title("%s" % np.exp(mcmc_result[-1]))
plt.savefig("%s/%s_prediction" % (DIR, ID))
print("%s/%s_prediction.png" % (DIR, ID))

# plot lc
x, y = np.genfromtxt("simulations/000{0}.txt".format(id)).T
plt.clf()
var = np.var(y)
y *= var
plt.plot(x, y, "k")
plt.xlim(0, 300)
plt.xlabel("$\mathrm{Time~(Days)}$")
plt.ylabel("$\mathrm{Normalised~Flux}$")
plt.subplots_adjust(left=.18)
plt.savefig("demo_lc.pdf")
