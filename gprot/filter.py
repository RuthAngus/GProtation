import numpy as np
from scipy.signal import butter, lfilter


def sigma_clip(x, y, yerr, nsigma):
    med = np.median(y)
    std = (sum((med - y)**2)/float(len(y)))**.5
    m = np.abs(y - med) > (nsigma * std)

    return x[~m], y[~m], yerr[~m]    

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def fill_gaps(x, y, yerr, cadence=1766./86400, make_uniform=True):
    # Guess cadence if not provided
    if cadence is None:
        cadence = np.median(np.diff(x))

    new_x = x.copy()
    new_y = y.copy()
    new_yerr = yerr.copy()
    
    # Find data gaps.
    dx = np.diff(x)
    i_gaps = np.where(dx > 1.5*cadence)[0]

    # Make gap filler arrays, using linear interpolation
    x_gaps = []
    y_gaps = []
    yerr_gaps = []
    for i in i_gaps:
        x0, x1 = x[i:i+2]
        y0, y1 = y[i:i+2]
        xfill = np.arange(x0 + cadence, x1, cadence)
        x_gaps.append(xfill)
        y_gaps.append(y0 + (xfill - x0)*(y1 - y0)/(x1 - x0))
        yerr_gaps.append(np.ones(len(xfill)) * yerr[i])
        
    # Insert gap fillers into gaps
    shift = 1
    i_new = []
    for i, xg, yg, yerrg in zip(i_gaps, x_gaps, y_gaps, yerr_gaps):
        ind = i + shift
        new_x = np.insert(new_x, ind, xg)
        new_y = np.insert(new_y, ind, yg)
        new_yerr = np.insert(new_yerr, ind, yerrg)
        n = len(xg)
        i_new.append(np.arange(ind, ind+n))
        shift += n
    
    if len(i_new) > 0:
        i_new = np.concatenate(i_new)
    else:
        i_new = np.array([])

    if make_uniform:
        # Regularize x to be exactly according to cadence
        new_x = np.arange(len(new_x))*cadence + new_x[0]
        
    return new_x, new_y, new_yerr, i_new

def bandpass_filter(x, y, yerr, pmin=0.5, pmax=100, cadence=1766./86400,
                    edge=2000, order=3):
    x, y, yerr, i_new = fill_gaps(x, y, yerr)

    # Sampling and cutoff frequencies
    fs = 1./cadence
    lowcut = 1./pmax
    highcut = 1./pmin

    yfilt = butter_bandpass_filter(y, lowcut, highcut, fs, order=order) 

    return (np.delete(x, i_new), 
            np.delete(yfilt, i_new), 
            np.delete(yerr, i_new))
