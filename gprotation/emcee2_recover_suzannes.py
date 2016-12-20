from __future__ import print_function
import numpy as np
import pandas as pd
from multiprocessing import Pool

import os

from emcee2_gprot_fit import gp_fit


def load_suzanne_lcs(id, DATA_DIR):
    sid = str(int(id)).zfill(4)
    x, y = np.genfromtxt(os.path.join(DATA_DIR,
                                      "lightcurve_{0}.txt".format(sid))).T
    return x - x[0], y - 1


def recover(i):

    RESULTS_DIR = "results_emcee2"  # just 2 sets of 200 days
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    DATA_DIR = "../code/simulations/kepler_diffrot_full/final"
    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0

    id = truths.N.values[m][i]
    sid = str(int(id)).zfill(4)
    print(id, i, "of", len(truths.N.values[m]))
    x, y = load_suzanne_lcs(sid, DATA_DIR)
    yerr = np.ones_like(y) * 1e-5

    gp_fit(x, y, yerr, sid, RESULTS_DIR)


if __name__ == "__main__":

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0

    pool = Pool()
    results = pool.map(recover, range(len(truths.N.values[m])))
