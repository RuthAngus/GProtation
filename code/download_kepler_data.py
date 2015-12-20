from __future__ import print_function
import numpy as np
import kplr
client = kplr.API()

data = np.genfromtxt("data/garcia.txt", skip_header=1).T
kid = data[0]
for i, id in enumerate(kid):
    print(i, "of", len(kid))
    star = client.star(str(int(id)))
    lc = star.get_light_curves(fetch=True, shortcadence=False)
    print(lc)
