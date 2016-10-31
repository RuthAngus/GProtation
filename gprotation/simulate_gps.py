from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
import os
import pandas as pd

P_DIR = "periodic_gp_simulations"
A_DIR = "aperiodic_gp_simulations"

nsims = 333

mus = np.array([-12, 7, -1, -17])
sigmas = np.array([5.4, 10, 3.8, 1.7])
# sigmas = np.array([1, 1, 1, 1])
periods = np.random.uniform(np.log(.5), np.log(100), nsims)

As = sigmas[0] * np.random.randn(nsims) + mus[0]
ls = sigmas[1] * np.random.randn(nsims) + mus[1]
gammas = sigmas[2] * np.random.randn(nsims) + mus[2]
ss = sigmas[3] * np.random.randn(nsims) + mus[3]

df = pd.DataFrame({"ln A": As, "ln l": ls, "ln gamma": gammas, "ln sigma": ss,
                   "ln period": periods})
df.to_csv("gp_truths.csv")

xs_l = np.arange(0, 200, .02043365)  # kepler cadence
xs = xs_l[::10]
for i, period in enumerate(periods):
    print(i, "of", len(periods))
    sid = str(int(i)).zfill(4)
    A, l, gamma = np.exp(As[i]), np.exp(ls[i]), np.exp(gammas[i])
    period, sigma = np.exp(periods[i]), np.exp(ss[i])
    kp = A * ExpSquaredKernel(l) \
            * ExpSine2Kernel(gamma, period)
    gp = george.GP(kp)
    ys = gp.sample(xs)
    y_noise = ys + np.random.randn(len(ys)) * 1e-5
    data = np.vstack((xs, ys))
    data_noise = np.vstack((xs, y_noise))
    np.savetxt(os.path.join(P_DIR, "{0}.txt".format(sid)), data.T)
    np.savetxt(os.path.join(P_DIR, "{0}_noise.txt".format(sid)), data_noise.T)

    k = A * ExpSquaredKernel(l)
    gp = george.GP(k)
    ys = gp.sample(xs)
    y_noise = ys + np.random.randn(len(ys)) * 1e-5
    data = np.vstack((xs, ys))
    data_noise = np.vstack((xs, y_noise))
    np.savetxt(os.path.join(A_DIR, "{0}.txt".format(sid)), data.T)
    np.savetxt(os.path.join(A_DIR, "{0}_noise.txt".format(sid)), data_noise.T)
