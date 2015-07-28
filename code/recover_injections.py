import numpy as np
import matplotlib.pyplot as plt
from measure_GP_rotation import fit
from Kepler_ACF import corr_run

def recover_injections():

    ids, true_ps, true_as = np.genfromtxt("simulations/true_periods.txt").T

    for i, id in enumerate(ids):
        x, y = np.genfromtxt("simulations/%s.txt" % int(id)).T
        yerr = np.ones_like(y) * 1e-5
        x -= x[0]
        cutoff = 100

        print "true period = ", true_ps[i]

        m = x < cutoff
#         plt.clf()
#         plt.errorbar(x[m], y[m], yerr=yerr[m], fmt="k.", capsize=0, ecolor=".7")
#         plt.show()

        # initialise with acf
        try:
            p_init = np.genfromtxt("simulations/%s_result.txt" % id)
        except:
            corr_run(x, y, yerr, id,
                     "/Users/angusr/Python/GProtation/code/simulations")
            p_init = np.genfromtxt("simulations/%s_result.txt" % id)
        print "acf period, err = ", p_init

        plims = [p_init[0]*.5, p_init[0]*2.]
        fit(x, y, yerr, int(id), p_init[0], plims, burnin=500, run=1500, npts=48,
                cutoff=cutoff, sine_kernel=False, acf=False)

if __name__ == "__main__":
    recover_injections()
