import numpy as np
import matplotlib.pyplot as plt
from measure_GP_rotation import fit
from Kepler_ACF import corr_run

def recover_injections():

    ids, true_ps, true_as = np.genfromtxt("simulations/true_periods.txt").T

    for i, id in enumerate(ids):
        x, y = np.genfromtxt("simulations/%s.txt" % int(id)).T
        yerr = np.ones_like(y) * 1e-5

        print "true period = ", true_ps[i]

        # initialise with acf
        try:
            p_init = np.genfromtxt("simulations/%s_result.txt" % id)
        except:
            corr_run(x, y, yerr, id,
                     "/Users/angusr/Python/GProtation/code/simulations")
            p_init = np.genfromtxt("simulations/%s_result.txt" % id)
        print "acf period, err = ", p_init

        plims = [p_init[0]*.7, p_init[0]*1.5]
        npts = int(p_init[0] / 10. * 48)  # 10 points per period
        cutoff = 10 * p_init[0]
        fit(x, y, yerr, str(int(id)).zfill(4), p_init[0], np.log(plims),
                burnin=1000, run=1500, npts=npts, cutoff=cutoff,
                sine_kernel=True, acf=False)

# def compare_truth(N=100):
#     truth = np.genfromtxt("simulations/true_periods.txt").T
#     acf_periods, GP_periods = [], []
#     for i in range(N):
#         print str(i).zfill(4)
#
#         acf = np.genfromtxt("simulations/%s.txt" float(i)).T
#         print acf
#
#         GP = np.genfromtxt("simulations/%s_result.txt" str(i).zfill(4)).T
#         print GP
#
#         assert 0
#         acf_periods.append(acf)

if __name__ == "__main__":
    recover_injections()
