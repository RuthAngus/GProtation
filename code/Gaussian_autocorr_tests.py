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


def evaluate_convergence(flat, mean_acorr):
    converged = False
    acorr_t = emcee.autocorr.integrated_time(flat, c=1)
    mean_acorr.append(np.mean(acorr_t))
    mean_ind = len(flat) / np.mean(acorr_t)
    mean_diff = None
    if len(mean_acorr) > 1:
        mean_diff = np.mean(mean_acorr[i] - mean_acorr[i-1])
        if mean_diff < 0 and mean_ind > 1000:
            converged = True
    return converged, mean_acorr, mean_ind, mean_diff


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
    nwalkers, ndim, nsteps = 12, 2, 100
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
    p0 = [theta_init + 1e-1 * np.random.rand(ndim) for i in range(nwalkers)]
    mean_acorr, mean_diff, mean_ind = [], [], []
    for i in range(40):
        p0, _, _ = sampler.run_mcmc(p0, nsteps)
        flat = np.reshape(sampler.chain, (nwalkers * nsteps*(i+1), ndim))
        c, mean_acorr, mean_i, mean_d = evaluate_convergence(flat, mean_acorr)
        mean_ind.append(mean_i)
        mean_diff.append(mean_d)
        print(c)
        if c:
            break


fig = corner.corner(flat)
fig.savefig("triangle")

plt.clf()
plt.plot(mean_ind)
plt.ylabel("Independent samples")
plt.xlabel("Steps")
plt.savefig("independent")

plt.clf()
plt.plot(mean_acorr)
plt.ylabel("Autocorrelation time")
plt.xlabel("Steps")
plt.savefig("autocorrelation_times")

plt.clf()
plt.plot(mean_diff)
plt.ylabel("Delta autocorrelation time")
plt.xlabel("Steps")
plt.savefig("diff")
