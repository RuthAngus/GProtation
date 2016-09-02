import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel


def generate_GP(theta, x, yerr):
    """
    Using the parameters provided in theta, generate a GP with x time sampling.
    theta: array of log(A), log(l), log(g), log(sigma), log(P).
    x: array of time values.
    yerr: array of y uncertainties
    """
    theta = np.exp(theta)
    k = theta[0] * ExpSquaredKernel(theta[1]) \
            * ExpSine2Kernel(theta[2], theta[4]) + WhiteKernel(theta[3])
    gp = george.GP(k, solver=george.HODLRSolver)
    gp.compute(x, np.sqrt(theta[3]+yerr**2))
    return gp.sample(x)


if __name__ == "__main__":
    x = np.arange(0, 20, .1)
    yerr = np.ones_like(x) * 1e-5

    theta = np.log([np.exp(-5), np.exp(7), np.exp(.6), np.exp(-16), 4])
    ys = generate_GP(theta, x, yerr)

    data = np.vstack((x, ys))
    np.savetxt("simulations/0000_gp.txt", data.T)

    plt.clf()
    plt.errorbar(x, ys, yerr=yerr, fmt="k.")
    plt.savefig("simulations/0000_gp")
