from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import h5py
import triangle
import george
from george.kernels import ExpSquaredKernel, ExpSine2Kernel, WhiteKernel
from plotstuff import colours
cols = colours()
from kepler_data import load_kepler_data
import glob

plotpar = {'axes.labelsize': 20,
           'xtick.labelsize': 16,
           'ytick.labelsize': 16,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def GP_demos(id, path):

    id = str(int(id)).zfill(4)
    with h5py.File("{0}/{1}_samples.h5".format(path, id), "r") as f:
        samples = f["samples"][...]
    nwalkers, nsteps, ndims = np.shape(samples)
    flat = np.reshape(samples, (nwalkers*nsteps, ndims))
    fig_labels = ["$\log(A)$", "$\log(l)$", "$\log(\Gamma)$", "$\log(\sigma)$",
                  "$P_{\mathrm{rot}}$"]
    flat[:, -1] = np.exp(flat[:, -1])
    fig = triangle.corner(flat, labels=fig_labels)
    plt.savefig("demo_triangle_{0}.pdf".format(id))
    flat[:, -1] = np.log(flat[:, -1])

    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(flat, [16, 50, 84], axis=0)))
    theta = np.exp(np.array([mcmc_result[i][0] for i in range(ndims)]))

    x, y = np.genfromtxt("{0}/lightcurve_{1}.txt".format(path, id)).T
    m = x < 300
    x, y = x[m], y[m]
    yerr = np.ones_like(y) * 1e-5
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], theta[4])
    gp = george.GP(k, solver=george.HODLRSolver)
    gp.compute(x, yerr)
    xs = np.linspace(x[0], x[-1], 1000)
    mu, cov = gp.predict(y, xs)

    plt.clf()
    var = np.var(y)
    # y *= var
    plt.plot(x, y, lw=5, color=".2", label="$\mathrm{Simulated~light~curve}$")
    plt.plot(xs, mu, color=cols.pink, label="$\mathrm{Best~fit~GP~model}$")
    plt.xlim(0, 300)
    plt.xlabel("$\mathrm{Time~(Days)}$")
    plt.ylabel("$\mathrm{Normalised~Flux}$")
    plt.subplots_adjust(left=.18, bottom=.12)
    plt.legend()
    plt.savefig("demo_lc_{0}.pdf".format(id))

def find_periodic(kid):

    kid = str(int(kid)).zfill(9)
    path = "/home/angusr/.kplr/data/lightcurves"
    fnames = glob.glob("{0}/{1}/*llc.fits".format(path, kid))
    x, y, yerr = load_kepler_data(fnames)
    x -= min(x)
    m = (x < 30) * (12 < x)
    x, y, yerr = x[m]-x[m][0], y[m], yerr[m]

    xs = np.linspace(min(x), max(x), 1000)
    theta = [1e-2, .1, np.exp(.6), 5, 1e-5**2]
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], \
            theta[3]) + WhiteKernel(theta[4])
    gp = george.GP(k, solver=george.HODLRSolver)
    gp.compute(x, yerr)
    mu, cov = gp.predict(y, xs)
    std = np.diag(cov**.5)

    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt="k.", ecolor=".7", capsize=0)
    plt.plot(xs, mu, color="CornflowerBlue", lw=3,
             label="$\mathrm{QP~kernel}$")
#     plt.fill_between(xs, mu-std, mu+std, color="CornflowerBlue", alpha=.5)

    theta = [1e-2, .1, 1e-5]
    k = theta[0] * ExpSquaredKernel(theta[1]) + WhiteKernel(theta[2])
    gp = george.GP(k, solver=george.HODLRSolver)
    gp.compute(x, yerr)
    semu, secov = gp.predict(y, xs)

    plt.plot(xs, semu, "--", color="DeepPink", lw=2,
             label="$\mathrm{SE~kernel}$")

    plt.xlabel("$\mathrm{Time~(Days)}$")
    plt.ylabel("$\mathrm{Normalised~(Flux)}$")
    plt.legend()
    plt.subplots_adjust(left=.18, bottom=.12)
    plt.savefig("klcs/{0}.pdf".format(kid))

if __name__ == "__main__":
    path = "../ruth/simulations"
    id = 97
    GP_demos(id, path)

#     kids = np.genfromtxt("kids.txt")
#     find_periodic(kids[1])
#     for kid in kids:
#         find_periodic(kid)
