# Code for generating periodic and non-periodic GP light curves.
# Find the appropriate parameters from the real light curves and THEN do this
# step.

# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel

SIM_DIR = "../simulations"

theta_init = np.log([np.exp(-5), np.exp(7), np.exp(.6), np.exp(-16),
                        p_init])

# Periodic GPs
# Periods log uniform from .5 to 100 days
periods = np.random.uniform(np.log(.5), np.log(100))

# Amplitudes log uniform from -6 to -3


# Gammas

# ls

# save truth file

# generate and save GPs.

# Non-Periodic GPs

# Amplitudes

# ls

# generate and save GPs.

# save truth file.
