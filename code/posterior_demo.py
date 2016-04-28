from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import h5py
import triangle
import george
from george.kernels import ExpSquaredKernel, ExpSine2Kernel
from plotstuff import colours
cols = colours()
from kepler_data import load_kepler_data
import glob

plotpar = {'axes.labelsize': 24,
           'xtick.labelsize': 16,
           'ytick.labelsize': 16,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def GP_demos(id):

    with h5py.File("sine/000{0}_samples.h5".format(id), "r") as f:
        samples = f["samples"][...]
    nwalkers, nsteps, ndims = np.shape(samples)
    flat = np.reshape(samples, (nwalkers*nsteps, ndims))
    fig_labels = ["$\log(A)$", "$\log(l)$", "$\log(\Gamma)$", "$\log(\sigma)$",
                  "$P_{\mathrm{rot}}$"]
    # flat[:, -1] = np.exp(flat[:, -1])
    # fig = triangle.corner(flat, labels=fig_labels)
    # plt.savefig("demo_triangle.pdf")

    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(flat, [16, 50, 84], axis=0)))
    theta = np.exp(np.array([mcmc_result[i][0] for i in range(ndims)]))

    x, y = np.genfromtxt("simulations/000{0}.txt".format(id)).T
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
    plt.savefig("demo_lc.pdf")

def find_periodic(kid):

    kid = str(int(kid)).zfill(9)
    path = "/home/angusr/.kplr/data/lightcurves"
    fnames = glob.glob("{0}/{1}/*llc.fits".format(path, kid))
    x, y, yerr = load_kepler_data(fnames)
    x -= min(x)
    m = x < 100
    x, y, yerr = x[m], y[m], yerr[m]

    theta = [np.exp(-5), np.exp(7), np.exp(.6), np.exp(-16), 10]
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], \
            theta[4])
    gp = george.GP(k, solver=george.HODLRSolver)
    gp.compute(x, yerr)
#     gp.optimize(x, y, yerr)
    xs = np.linspace(x[0], x[-1], 1000)
    mu, cov = gp.predict(y, xs)

    plt.clf()
    plt.plot(x, y, "k.")
    plt.plot(xs, mu)
    plt.savefig("klcs/{0}.pdf".format(kid))

if __name__ == "__main__":
    kids = np.genfromtxt("kids.txt")
    for kid in kids:
        find_periodic(kid)
