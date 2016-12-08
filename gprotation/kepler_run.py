# measure rotation periods for stars in the McQuillan catalogue
from __future__ import print_function
import numpy as np
import subprocess
import pandas as pd
import os
import gprot_fit as gp
import kplr
client = kplr.API()
import kepler_data as kd
import matplotlib.pyplot as plt
from multiprocessing import Pool


def run_gprot(stars):
    for i, star in enumerate(stars):
        print(int(star), i, "of", len(stars))
        print("gprot-fit {0} --kepler -o -v".format(int(star)))
        subprocess.call("gprot-fit {0} --kepler -o -v".format(int(star)),
                        shell=True)


def run_my_fit(i):
    ruth1 = "/export/bbq1/angusr/GProtation/"
    lc_path = "/home/angusr/.kplr/data/lightcurves/"
    RESULTS_DIR = "/export/bbq1/angusr/GProtation/gprotation/kepler"

    # McQuillan stars
    s = pd.read_csv(os.path.join(ruth1, "code/data/Table_1_Periodic.txt"))
    kid = s.KIC.values[i]

    print(int(kid), i, "of", len(stars))
    star = client.star(int(kid))
    lcs = star.get_light_curves(fetch=True, short_cadence=False,
                                clobber=False)
    LC_DIR = os.path.join(lc_path, "{}".format(str(int(kid)).zfill(9)))
    x, y, yerr = kd.load_kepler_data(LC_DIR)

    emcee2 = False
    if emcee2:
        e2.gp_fit(x, y, yerr, "{}2".format(id), RESULTS_DIR)

    fit = gp.fit(x, y, yerr, kid, RESULTS_DIR)
#     fit.gp_fit(burnin=1000, nwalkers=16, nruns=5, full_run=1000, nsets=2)
    fit.gp_fit(burnin=2, nwalkers=12, nruns=2, full_run=50, nsets=2)  # fast


if __name__ == "__main__":
    # NGC6891
    ruth2 = "/export/bbq1/angusr/GProtation/"
    stars = np.genfromtxt(os.path.join(ruth2,
                          "code/data/NGC6819_members.txt")).T

    for i in range(2):
        run_my_fit(i)

#     pool = Pool()
#     pool.map(run_my_fit, range(2))
#     pool.map(run_my_fit, range(100))
