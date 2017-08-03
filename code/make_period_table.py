"""
Make a table of the KOI rotation periods.
"""

import os
import numpy as np
import pandas as pd
import glob
import kplr
import matplotlib.pyplot as plt
import teff_bv as tbv


def get_KOI_names():
    # client = kplr.API()
    koi_list = np.sort(glob.glob(os.path.join(RESULTS_DIR, "*h5")))
    kois, kepids, mass, teff, logg = [], [], [], [], []
    for i, koi_name in enumerate(koi_list):
        koi = int((koi_name.split("/")[-1].split("-")[-1].split(".")[0]))
        kois.append(koi)
        # koi = client.koi(952.01)
        # star = client.koi("{}.01".format(koi))
        # kepids.append(star.kepid)
    return np.sort(kois)


def get_period_medians(koi_names):
    recovered, errp, errm, lnerrp, lnerrm = [], [], [], [], []
    for i, kid in enumerate(koi_names):
        fn = os.path.join(RESULTS_DIR, "KOI-{}.h5".format(int(kid)))
        if os.path.exists(fn):
            df = pd.read_hdf(fn, key="samples")
            samps = np.exp(df.ln_period.values)
            recovered.append(np.median(samps))
            upper = np.percentile(samps, 84)
            lower = np.percentile(samps, 16)
            errp.append(upper - recovered[i])
            errm.append(recovered[i] - lower)
    return recovered, np.array(errp), np.array(errm)


def get_kic_properties(kois):
    df1 = pd.read_csv("cumulative.csv", skiprows=155)

    k = []
    for i, koi in enumerate(kois):
        k.append("K{}.01".format(str(koi).zfill(5)))

    koi_df = pd.DataFrame({"kepoi_name": k})
    df = pd.merge(df1, koi_df)

    teff, teff_errp, teff_errm = df.koi_steff.values, \
        df.koi_steff_err1.values, df.koi_steff_err2.values
    logg, logg_errp, logg_errm = df.koi_slogg.values, \
        df.koi_slogg_err1.values, df.koi_slogg_err2.values
    feh, feh_errp, feh_errm = df.koi_smet.values, df.koi_smet_err1.values, \
        df.koi_smet_err2.values

    return teff, teff_errp, teff_errm, logg, logg_errp, logg_errm, feh, \
        feh_errp, feh_errm


def make_tex_table(kois, p, errp, errm, teff, teff_errp, teff_errm, logg,
                   logg_errp, logg_errm, feh, feh_errp, feh_errm):
    with open("period_table.tex", "w") as f:
        for i, koi in enumerate(kois[:10]):
            f.write("{0} & {1:.0f}$^{{+{2:.1f}}}_{{-{3:.1f}}}$ & "
                    "${4}^{{+{5}}}_{{{6}}}$ & ${7:.2f}^{{+{8:.2f}}}_{{{9:.2f}}}$ & "
                    "${10}^{{+{11:.1g}}}_{{{12:.1g}}}$"
                    "\\\ \n"
                    .format(koi, p[i], errp[i], errm[i], int(teff[i]),
                            int(teff_errp[i]), int(teff_errm[i]), logg[i], logg_errp[i],
                            logg_errm[i], feh[i], feh_errp[i], feh_errm[i]))
    f.close()


def make_data_file(kois, periods, errp, errm, teff, teff_errp, teff_err, logg,
                   logg_errp, logg_errm, feh, feh_errp, feh_errm):

    df = pd.DataFrame({"KOI": kois, "period": periods, "period_errp": errp,
                       "period_errm": errm, "teff": teff, "teff_errp":
                       teff_errp, "teff_errm": teff_errm, "logg": logg,
                       "logg_errp": logg_errp, "logg_errm": logg_errm, "feh":
                       feh, "feh_errp": feh_errp, "feh_errm": feh_errm})
    df.to_csv("koi_periods.csv")


def read_data_file():
    data = pd.read_csv("koi_periods.csv")
    return data.KOI.values, data.period.values, data.period_errp.values, \
        data.period_errm.values, data.teff.values, data.teff_errp.values, \
        data.teff_errm.values, data.logg.values, data.logg_errp.values, \
        data.logg_errm.values, data.feh.values, data.feh_errp.values, \
        data.feh_errm.values


if __name__ == "__main__":
    path = "/Users/ruthangus/projects/GProtation"
    RESULTS_DIR = os.path.join(path, "code/koi_results_01_23")

    kois = get_KOI_names()
    periods, errp, errm = get_period_medians(kois)
    plt.clf()
    plt.hist(errp, 100, alpha=.5)
    plt.hist(errm, 100, alpha=.5)
    plt.savefig("err_hist")
    plt.clf()
    plt.hist(periods)
    plt.savefig("phist")

    teff, teff_errp, teff_errm, logg, logg_errp, logg_errm, feh, feh_errp, \
        feh_errm = get_kic_properties(kois)

    periods = np.array(periods)
    m = periods > 1
    print(len(periods), "periods")

    teff, teff_errp, teff_errm, logg, logg_errp, logg_errm, feh, feh_errp, \
        feh_errm = teff[m], teff_errp[m], teff_errm[m], logg[m], \
        logg_errp[m], logg_errm[m], feh[m], feh_errp[m], feh_errm[m]
    kois, periods = np.array(kois)[m], periods[m]
    errp, errm = errp[m], errm[m]

    make_data_file(kois, periods, errp, errm, teff, teff_errp, teff_errm,
                   logg, logg_errp, logg_errm, feh, feh_errp, feh_errm)
    kois, periods, errp, errm, teff, teff_errp, teff_err, logg, logg_errp, \
        logg_errm, feh, feh_errp, feh_errm = read_data_file()
    make_tex_table(kois, periods, errp, errm, teff, teff_errp, teff_err, logg,
                   logg_errp, logg_errm, feh, feh_errp, feh_errm)

    plt.clf()
    plt.errorbar(tbv.teff2bv(teff, logg, feh), periods, yerr=[errp, errm],
                 fmt="k.", alpha=.5)
    plt.savefig("p_vs_bv")
