from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import glob
from ACF import load_fits
import emcee
import triangle

from params import plot_params
reb = plot_params()
from colours import plot_colours
cols = plot_colours()

# take x, y, yerr, initial guess and fit an initial GP
# prompt for initial guess
# return GP params
def basic(x, y, yerr, p_init, ID):

    print(ID)

    # initialise
    theta_init = [1, 1, 1, p_init]
    k = theta_init[0] * ExpSquaredKernel(theta_init[1]) \
            * ExpSine2Kernel(theta_init[2], theta_init[3])
    gp = george.GP(k)
    gp.compute(x, yerr)

    # optimise
    theta = gp.optimize(x, y, yerr)[0]

    # predict
    xs = np.linspace(x[0], x[-1], 1000)
    mu, cov = gp.predict(y, xs)

    plt.clf()
    plt.errorbar(x, y, yerr=yerr, **reb)
    plt.plot(xs, mu, color=cols.blue)
    plt.savefig("%s_predict" % ID)

    return theta

# model
def model(theta, x, y, yerr):
    k = theta[0] * ExpSquaredKernel(theta[1]) \
            * ExpSine2Kernel(theta[2], theta[3])
    gp = george.GP(k)
    gp.compute(x, yerr)

    # predict
    mu, cov = gp.predict(y, x)
    return mu, cov

# lnprior
def lnprior(theta):
    if 1e-10 < theta[0] < 1e10 and 1e-10 < theta[1] < 1e10 and \
            1e-10 < theta[2] < 1e10 and 0 < theta[3] < 100:
        return 0.0
    return -np.inf

# lnprob
def lnprob(theta, x, y, yerr):
    return lnlike(theta, x, y, yerr) + lnprior(theta)

# lnlike
def lnlike(theta, x, y, yerr):
    k = theta[0] * ExpSquaredKernel(theta[1]) \
            * ExpSine2Kernel(theta[2], theta[3])
    gp = george.GP(k)
    try:
        gp.compute(x, np.sqrt(theta[4]+yerr**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    return gp.lnlikelihood(y, quiet=True)

# take x, y, yerr and initial guess and do MCMC
def MCMC(theta_init, x, y, yerr, ID):

    ndim, nwalkers = len(theta_init), 32
    p0 = [theta_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (x, y, yerr)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)

    print("burning in...")
    p0, lp, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    p0, lp, state = sampler.run_mcmc(p0, 100)

    flat = sampler.chain[:, 50:, :].reshape((-1, ndim))
    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(flat, [16, 50, 84], axis=0)))

    print("Plotting traces")
    for i in range(ndim):
        plt.clf()
        plt.axhline(theta_init[i], color = "r")
        pl.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3)
        pl.savefig("%s.png" % i)

    print("Making triangle plot")
    fig_labels = ["$A$", "$l_2$", "$l_1$", "$s$", "$P$"]
    fig = triangle.corner(sampler.flatchain, truths=theta_init,
                          labels=fig_labels)
    fig.savefig("triangle.png")

if __name__ == "__main__":

    D = "/Users/angusr/angusr/data2/Q15_public"
    fnames = glob.glob("%s/kplr0081*" % D)

    for fname in fnames:
        x, y, yerr = load_fits(fname)
        l = np.isfinite(x) * np.isfinite(y) * np.isfinite(yerr)
        x, y, yerr = x[l], y[l], yerr[l]
        kid = fname[42:51]

        # subsample
        s = 10
        x, y, yerr = x[::s], y[::s], yerr[::s]

        # median normalise
        med = np.median(y)
        y /= med
        yerr /= med

        # try optimising
#         p_init = float(raw_input("Enter initial guess for rotation period "))
#         theta = basic(x, y, yerr, p_init, kid)
#         print(theta)
#         choose = raw_input("Use these parameters? y/n ")
#         if choose == "y": theta = theta
#         else: theta = [1., 1., 1., p_init]

        theta = [1., 1., 1., 8., 1.]
        MCMC(np.array(theta), x, y, yerr, kid)
