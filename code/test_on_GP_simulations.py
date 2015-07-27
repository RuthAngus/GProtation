import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, CosineKernel

def generate(pars, sine_kernel=True):
    '''
    A function to generate a set of test data.
    '''
    print "pars = ", pars

    x = np.arange(0, 10, .1)
    yerr = np.ones_like(x) * .001

    if sine_kernel:
        k = pars[0] * ExpSquaredKernel(pars[1]) \
                * ExpSine2Kernel(pars[2], pars[4])
    else:
        k = pars[0] * ExpSquaredKernel(pars[1]) \
                * CosineKernel(pars[3])
    gp = george.GP(k)
    gp.compute(x, yerr)  # no added noise
    ys = gp.sample(x, 1) + np.random.randn(len(x))*yerr

    plt.clf()
    plt.plot(x, ys, "k.")
    mu, cov = gp.predict(ys, x)
    plt.plot(x, mu)
    plt.savefig("test_data")
    return x, ys, yerr

def fit(theta_init, x, ys, yerr, plims):
    """
    fitting the GP. theta_init should be in linear space, not log.
    as should plims
    """
    theta_init = np.log(theta_init)
    DIR = "../figs"
    sampler = MCMC(theta_init, x, ys, yerr, plims, 200, 500, "test", DIR)
    make_plot(sampler, x, ys, yerr, 1, DIR, traces=True)

if __name__ == "__main__":
    # generate a fake lc and see if you can recover it
    plims = [2., 30.]
    sine_kernel = False

    if sine_kernel:
        pars = [1., 1., 1., 1., 7.]
        theta_init = [1., 1., 1., 1., 7.]
        from GProtation import MCMC, make_plot
    else:
        pars = [1., 1., 1., 7.]
        theta_init = [1., 1., 1., 7.]
        from GProtation_cosine import MCMC, make_plot

    x, y, yerr = generate(pars, sine_kernel=False)
    fit(theta_init, x, y, yerr, plims)
