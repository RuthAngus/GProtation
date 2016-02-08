import numpy as np
import matplotlib.pyplot as plt

def compare_acf(true_periods, ids, path):

    # load recovered
    recovered_periods = np.zeros_like(ids)
    errs = np.zeros_like(ids)
    for i in range(len(ids)):
        id = str(int(ids[i])).zfill(4)
        recovered_periods[i], errs[i] = \
                np.genfromtxt("{0}/{1}_acfresult.txt".format(path, id)).T

    plt.clf()
    recovered_periods[recovered_periods==-9999] = 0
    plt.plot(true_periods, recovered_periods, "k.")
    plt.ylim(0, 80)
    xs = np.linspace(min(true_periods), max(true_periods), 100)
    plt.plot(xs, xs, "r--")
    plt.savefig("acf_compare")

if __name__ == "__main__":

    # Load Suzanne's noise-free simulations
    data = np.genfromtxt("../par/final_table.txt", skip_header=1).T
    m = data[13] == 0  # just the stars without diffrot
    ids = data[0][m]
    true_periods = data[-3][m]

#     # my noise-free simulations
#     ids, true_periods, true_amps = \
#             np.genfromtxt("../../noise-free/true_periods_amps.txt").T
#     compare(true_periods, ids, "../../noise-free")
    compare(true_periods, ids, "noise-free")
