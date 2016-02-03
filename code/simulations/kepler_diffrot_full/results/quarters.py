import numpy as np


def split_into_quarters(x, y, yerr):
    """
    This function takes x, y, and yerr and, using a file containing the
    start-time of each Kepler quarter, splits the data into a list of
    lists of separate quarters.
    """
    _, qs = np.genfromtxt("quarters.txt", skip_header=1).T
    qs -= qs[0]
    newx, newy, newyerr = [], [], []
    for i in range(len(qs)-1):
        m = (qs[i] < x) * (x < qs[i+1])
        if len(x[m]):  # only if that quarter exists
            newx.append(x[m])
            newy.append(y[m])
            newyerr.append(yerr[m])
        m = (x > qs[1])
        newx.append(x[m])
        newy.append(y[m])
        newyerr.append(yerr[m])
        return newx, newy, newyerr


def lnprob_split(theta, xb, yb, yerrb, plims):
    """
    A log-probability function that takes the model parameters, lists of lists
    of xb, yb, yerrb (one for every quarter) and the plims for the period
    prior.
    theta must be a list in the following order, t = [A, l, g, s, p]
    """
    nqs = len(xb)
    lps = [lnlike(theta, np.array(xb[i]), np.array(yb[i]), np.array(yerrb[i]))
           for i in range(nqs)]
    return np.sum(lps) + lnprior(t, plims)
