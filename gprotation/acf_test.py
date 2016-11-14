# checking whether the ACF underestimates are produced by uneven sampling.
import numpy as np
import matplotlib.pyplot as plt
import Kepler_ACF as ka


def gen_data(period):
    x = np.arange(0, 200, .02043365)
    return x, np.sin(2 * np.pi * x / period)


def make_gaps(x, y):
    m = np.ones(len(x), dtype=bool)
    l = np.random.choice(np.arange(len(x)), 20)
    for i in l:
        m[i] = False
    return x[m], y[m]


if __name__ == "__main__":

    N = 10
    periods = np.exp(np.random.uniform(-1, 4.6, N))
    p1, p2 = [np.zeros(N) for i in range(2)]
    for i, period in enumerate(periods):
        print(i, "of", N)
        x, y = gen_data(period)
        x_gaps, y_gaps = make_gaps(x, y)
        period1, err1, lags1, acf1 = ka.corr_run(x, y, np.ones_like(y)*1e-5,
                                                 "test", ".")
        period2, err2, lags2, acf2 = ka.corr_run(x_gaps, y_gaps,
                                                 np.ones_like(y_gaps)*1e-5,
                                                 "test", ".")
        print(period1, period2)
        p1[i] = period1
        p2[i] = period2

    plt.clf()
    plt.plot(periods, p1, "b.")
    plt.plot(periods, p2, "r.")
    plt.savefig("test_compare")
