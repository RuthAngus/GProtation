import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def MAD(true, y):
    return np.median(np.abs(y - true))


def MAD_rel(true, y):
    return np.median(np.abs(y - true)/true) * 100


def RMS(true, y):
    return (np.mean(true - y)**2)**.5


def load_GP_results(prior=True):
    nbins = 95
    if prior:
        RESULTS_DIR = "results_acfprior_03_10"
    else:
        RESULTS_DIR = "results_noprior_03_10"
    truths = pd.read_csv("final_table.txt", delimiter=" ")

    # remove differential rotators and take just the first 100
    m = truths.DELTA_OMEGA.values == 0
    truths = truths.iloc[m]

    recovered = np.zeros(len(truths.N.values))
    errp, errm = [np.zeros(len(truths.N.values)) for i in range(2)]
    lnerrp, lnerrm = [np.zeros(len(truths.N.values)) for i in range(2)]
    for i, id in enumerate(truths.N.values):
        fn = os.path.join(RESULTS_DIR, "{}.h5".format(id))
        if os.path.exists(fn):
            df = pd.read_hdf(fn, key="samples")
            phist, bins = np.histogram(df.ln_period.values, nbins)
            ln_p = bins[phist == max(phist)][0]
            recovered[i] = np.exp(ln_p)
            lnerrp[i] = np.percentile(df.ln_period.values, 84) - ln_p
            lnerrm[i] = ln_p - np.percentile(df.ln_period.values, 16)
            errp[i] = np.exp(lnerrp[i]/ln_p)
            errm[i] = np.exp(lnerrm[i]/ln_p)
    return truths.P_MIN.values, recovered, errp, errm


def load_acf_pgram_results():
    truths = pd.read_csv("truths_extended_02_17.csv")
    return truths.P_MIN.values, truths.acf_period.values, \
        truths.pgram_period.values


def load_tel_aviv_results():
    data = pd.read_csv("telaviv_acf_output.txt")
    return data.period.values, data.period_err.values


if __name__ == "__main__":
    # Load Tel Aviv ACF results
    tlv_p, tlv_perr = load_tel_aviv_results()
    truths = pd.read_csv("final_table.txt", delimiter=" ")
    m = truths.DELTA_OMEGA.values == 0
    tlv_p = tlv_p[m]

    # Load Pgram results
    _, _, pgram = load_acf_pgram_results()

    # Load GP results
    true, gp_p1, gp_perrp1, gp_perrm1 = load_GP_results(prior=True)
    _, gp_p2, gp_perrp2, gp_perrm2 = load_GP_results(prior=False)

    plt.clf()
    plt.plot(true, pgram, "k.")
    plt.savefig("test_pgr")

    plt.clf()
    plt.plot(true, tlv_p, "k.")
    plt.savefig("test_tlv")

    plt.clf()
    plt.plot(true, gp_p1, "k.")
    plt.savefig("test_gp1")

    plt.clf()
    plt.plot(true, gp_p2, "k.")
    plt.savefig("test_gp2")

    print("Tel aviv RMS:", RMS(true, tlv_p))
    print("Tel aviv MAD:", MAD(true, tlv_p))
    print("Tel aviv MRD:", MAD_rel(true, tlv_p), "\n")

    print("pgram RMS:", RMS(true, pgram))
    print("pgram MAD:", MAD(true, pgram))
    print("pgram MRD:", MAD_rel(true, pgram), "\n")

    print("GP (prior) RMS:", RMS(true, gp_p1))
    print("GP (prior) MAD:", MAD(true, gp_p1))
    print("GP (prior) MRD:", MAD_rel(true, gp_p1), "\n")

    print("GP (no prior) RMS:", RMS(true, gp_p1))
    print("GP (no prior) MAD:", MAD(true, gp_p1))
    print("GP (no prior) MRD:", MAD_rel(true, gp_p1), "\n")
