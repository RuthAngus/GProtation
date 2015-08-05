import numpy as np
import matplotlib.pyplot as plt
from simple_acf import simple_acf
from Kepler_ACF import corr_run

def simulate(N):

    periods = np.random.uniform(50, 100, N)
    xs = np.arange(0, 1600, .02043365)
    xs = np.arange(0, 1600, .02043365)

    acf_ps, my_acf_ps = [], []
    for i, p in enumerate(periods):
        print "\n", i, "of", N, "\n"
#         ys = np.sin(xs * 2 * np.pi * (1./p)) + np.random.randn(len(xs))*.2
        ys = np.sin(xs * 2 * np.pi * (1./p))
#         period, acf, lags, flag = simple_acf(xs, ys)
        acfp, p_err = corr_run(xs, ys, 1e-5, i, "tests", saveplot=True)
        acf_ps.append(acfp[0])
#         my_acf_ps.append(period)

    plt.clf()
    xs = np.arange(0, 100, .1)
    plt.plot(periods, acf_ps, "r.")
#     plt.plot(periods, my_acf_ps, "b.")
    plt.plot(xs, xs, "k--")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()
    plt.savefig("sine_test")

    acf_ps = np.array(acf_ps)
#     my_acf_ps = np.array(my_acf_ps)
#     results = np.vstack((periods, acf_ps, my_acf_ps))
    results = np.vstack((periods, acf_ps))
    np.savetxt("sine_test.txt", np.transpose((results)))

if __name__ == "__main__":
    simulate(50)

#     data = np.genfromtxt("sine_test.txt")
#     periods, acf_ps, my_acf_ps = data

#     plt.clf()
#     xs = np.arange(0, 100, .1)
#     plt.plot(periods, acf_ps, "r.")
#     plt.plot(periods, my_acf_ps, "b.")
#     plt.plot(xs, xs, "k--")
#     plt.xlim(0, 100)
#     plt.ylim(0, 100)
#     plt.show()
