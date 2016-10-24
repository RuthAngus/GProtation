
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from GProtation import make_plot, lnprob
from Kepler_ACF import corr_run
import h5py
from gatspy.periodic import LombScargle
import sys
import os
import time
import emcee
import pyfits


# In[2]:

DATA_DIR = "data/"
RESULTS_DIR = "results/"


# In[3]:

def load_data(epic_id, DATA_DIR):
    hdulist = pyfits.open(os.path.join(DATA_DIR, "hlsp_everest_k2_llc_{0}-c04_kepler_v1.0_lc.fits".format(epic_id)))
    time, flux = hdulist[1].data["TIME"], hdulist[1].data["FLUX"]
    out = hdulist[1].data["OUTLIER"]
    m = np.isfinite(time) * np.isfinite(flux) * (out < 1)
    med = np.median(flux[m])
    return time[m], flux[m]/med - 1


# Load first target.

# In[4]:

epic_id = 211000411
#epic_id = 211098454
x, y = load_data(epic_id, DATA_DIR)
yerr = np.ones_like(y) * 1e-5
plt.plot(x, y, "k.")


# Calculate ACF using McQuillan (2013) code

# In[5]:

fname = os.path.join(RESULTS_DIR, "{0}_acf_result.txt".format(epic_id))
p_init, err, lags, acf = corr_run(x, y, yerr, epic_id, RESULTS_DIR)
np.savetxt(os.path.join(RESULTS_DIR, "{0}_acf_result.txt".format(epic_id)), np.array([p_init[0], err[0]]).T)
print("acf period, err = ", p_init, err)
plt.plot(lags, acf)


# Calculate periodogram

# In[6]:

ps = np.arange(.1, 100, .1)
print(type(x), type(y), type(yerr))
model = LombScargle().fit(x, y, yerr)
pgram = model.periodogram(ps)
plt.plot(ps, pgram)


# Prep MCMC and set limits on prior.

# Subsample the light curve.

# In[10]:

xb, yb, yerrb = x[::10], y[::10], yerr[::10]
plt.plot(xb, yb, "k.")


# In[17]:

plims = np.log([.1*p_init, 5*p_init])
print("total number of points = ", len(xb))
theta_init = np.log([np.exp(-5), np.exp(7), np.exp(.6), np.exp(-16), p_init])
burnin, nwalkers = 500, 12
runs = np.zeros(10) + 500
ndim = len(theta_init)
p0 = [theta_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
args = (xb, yb, yerrb, plims)


# Time the LHF call.

# In[18]:

start = time.time()
print("lnprob = ", lnprob(theta_init, x, y, yerr, plims))
end = time.time()
tm = end - start
print("1 lhf call takes ", tm, "seconds")
print("burn in will take", tm * nwalkers * burnin, "s")
print("each run will take", tm * nwalkers * runs[0]/60, "mins")
print("total = ", (tm * nwalkers * np.sum(runs) + tm * nwalkers * burnin)/60, "mins")


# Run MCMC.

# In[19]:

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
print("burning in...")
p0, _, state = sampler.run_mcmc(p0, burnin)

sample_array = np.zeros((nwalkers, sum(runs), ndim))
for i, run in enumerate(runs):
    sampler.reset()
    print("production run, {0} steps".format(int(run)))
    start = time.time()
    p0, _, state = sampler.run_mcmc(p0, run)
    end = time.time()
    print("time taken = ", (end - start)/60, "minutes")

    # save samples
    sample_array[:, sum(runs[:i]):sum(runs[:(i+1)]), :] = np.array(sampler.chain)
    f = h5py.File(os.path.join(RESULTS_DIR, "{0}.h5".format(epic_id)), "w")
    data = f.create_dataset("samples", np.shape(sample_array[:, :sum(runs[:(i+1)]), :]))
    data[:, :] = sample_array[:, :sum(runs[:(i+1)]), :]
    f.close()

    # make various plots
    with h5py.File(os.path.join(RESULTS_DIR, "{0}.h5".format(epic_id)), "r") as f:
        samples = f["samples"][...]
    mcmc_result = make_plot(samples, x, y, yerr, epic_id, RESULTS_DIR, traces=True, tri=True, prediction=True)


# Cut out burn in.

# In[22]:

with h5py.File(os.path.join(RESULTS_DIR, "{0}.h5".format(epic_id)), "r") as f:
    samples = f["samples"][...]
samps = samples[:, 300:, :]
mcmc_result = make_plot(samps, x, y, yerr, epic_id, RESULTS_DIR, traces=True, tri=True, prediction=True)


# In[ ]:



