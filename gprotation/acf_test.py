# checking whether the ACF underestimates are produced by uneven sampling.
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import Kepler_ACF as ka
import simple_acf as sa


def gen_data(period):
    x = np.arange(0, 200, .02043365)
    return x, np.sin(2 * np.pi * x / period)


def make_gaps(x, y, ngaps):
    m = np.ones(len(x), dtype=bool)
    l = np.random.choice(np.arange(len(x)), ngaps)
    for i in l:
        m[i] = False
    return x[m], y[m]


if __name__ == "__main__":

    ntests = 100
    amy = False
    periods = np.exp(np.random.uniform(-1, 4.6, ntests))
    p1, p2 = [np.zeros(ntests) for i in range(2)]
    for i, period in enumerate(periods):
        print(i, "of", ntests)
        x, y = gen_data(period)
        x_gaps, y_gaps = make_gaps(x, y, 1000)
        if amy:
            period1, err1, lags1, acf1 = ka.corr_run(x, y,
                                                     np.ones_like(y)*1e-5,
                                                     "test", ".")
            period2, err2, lags2, acf2 = ka.corr_run(x_gaps, y_gaps,
                                                     np.ones_like(y_gaps)*1e-5,
                                                     "test", ".")
        else:
            period1, acf, lags, rvar = sa.simple_acf(x, y)
            period2, acf, lags, rvar = sa.simple_acf(x_gaps, y_gaps)

        print(period1, period2)
        p1[i] = period1
        p2[i] = period2

    m = (p1 > 0) * (p2 > 0)
    plt.clf()
    plt.plot(np.log(periods[m]), np.log(p1[m]), "b.")
    plt.plot(np.log(periods[m]), np.log(p2[m]), "r.")
    plt.savefig("test_compare_simple")
