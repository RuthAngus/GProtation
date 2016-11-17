import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner


def lnlike(pars, x, y, yerr):
    model = Gauss(x, pars[0], pars[1])
    inv_sigma2 = 1./yerr**2
    return - .5 * (np.sum((y-model)**2 * inv_sigma2 - np.log(inv_sigma2)))


def lnprob(pars, x, y, yerr):
    if pars[1] < 0:
        return -np.inf
    return lnlike(pars, x, y, yerr)


def Gauss(x, mu, sigma):
    return np.exp(-.5 * ((x - mu)**2/(.5 * sigma**2)))


if __name__ == "__main__":

    x = np.arange(-10, 10, .1)
    y = Gauss(x, 0, 1) + np.random.randn(len(x)) * 1e-1
    y += Gauss(x, 2, 1)
    yerr = np.ones_like(y) * 1e-1

    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt="k.", capsize=0, ecolor=".7")
    plt.plot(x, Gauss(x, 0, 1))
    plt.savefig("test")

    args = (x, y, yerr)
    theta_init = [0, 1]
    nwalkers, ndim, nsteps = 12, 2, 1000
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
    p0 = [theta_init + 1e-1 * np.random.rand(ndim) for i in range(nwalkers)]
    acorrs, ind_mu, ind_sig = [], [], []
    for i in range(10):
        p0, _, _ = sampler.run_mcmc(p0, nsteps)

        flat = np.reshape(sampler.chain, (nwalkers * nsteps*(i+1), ndim))
        fig = corner.corner(flat)
        fig.savefig("triangle")

        acorr_t = emcee.autocorr.integrated_time(flat)
        ind_mu.append(len(flat) / acorr_t[0])
        ind_sig.append(len(flat) / acorr_t[1])
        print("independent samples mu = ", ind_mu[i])
        print("independent samples sig = ", ind_sig[i])
        print("acorr time = ", acorr_t)
        acorrs.append(acorr_t)

plt.clf()
plt.plot(ind_mu)
plt.plot(ind_sig)
plt.savefig("independent")

plt.clf()
plt.plot(acorrs)
plt.savefig("autocorrelation_times")
