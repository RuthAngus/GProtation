import numpy as np
import kplr
client = kplr.API()

ids = np.genfromtxt("NGC6819_members.txt", dtype=str).T
for id in ids:
    print id
    star = client.star(id)
    star.get_light_curves(fetch=True, short_cadence=False)
