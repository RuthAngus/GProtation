import numpy as np
import matplotlib.pyplot as plt
import sys
id = sys.argv[1]

x, y = np.genfromtxt("simulations/%s.txt" % id).T
plt.clf()
plt.plot(x[:300], y[:300], "k.")
plt.savefig("test")
