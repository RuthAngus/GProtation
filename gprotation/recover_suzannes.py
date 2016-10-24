import numpy as np
from GProt import calc_p_init, fit
import pandas as pd
import os

DATA_DIR = "../code/simulations/kepler_diffrot_full/final"
RESULTS_DIR = "results/"


def load_suzanne_lcs(id):
    id = str(int(id)).zfill(4)
    x, y = np.genfromtxt(os.path.join(DATA_DIR,
                                      "lightcurve_{0}.txt".format(id))).T
    return x - x[0], y - 1

if __name__ == "__main__":

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0

    for i, id in enumerate(truths.N.values[m]):  # zero diffrot
        print(id, i, "of", len(truths.N.values[m]))
        x, y = load_suzanne_lcs(str(int(id)).zfill(2))
        yerr = np.ones_like(y) * 1e-5
        acf_period, a_err, pgram_period, p_err = calc_p_init(x, y, yerr,
                                                             str(int(id))
                                                             .zfill(2))
        p_init = acf_period
        if acf_period == 0:
            p_init = pgram_period

        c = 200  # cut off at 200 days
        m = x < 200
        xb, yb, yerrb = x[m][::10], y[m][::10], yerr[m][::10]
        fit(xb, yb, yerrb, acf_period, str(int(id)).zfill(4))
