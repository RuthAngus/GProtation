import numpy as np
import matplotlib.pyplot as plt
import glob

files = glob.glob("simulations/????.txt")

plt.clf()
for i, f in enumerate(files[:3]):
    x, y = np.genfromtxt(f).T
    plt.plot(x, y, color=cols[i])
