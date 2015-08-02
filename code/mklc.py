import numpy as np
import scipy
import scipy.io
import pylab
import numpy
import glob
import pyfits

def mklc(t, nspot=200, incl=(scipy.pi)*5./12., amp=1., tau=30.5, p=10.0):

    diffrot = 0.

    ''' This is a simplified version of the class-based routines in
    spot_model.py. It generates a light curves for dark, point like
    spots with no limb-darkening.

    Parameters:
    nspot = desired number of spots present on star at any
            one time
    amp = desired light curve amplitude
    tau = characteristic spot life-time
    diffrot = fractional difference between equatorial and polar
              rotation period
    (unit of time is equatorial rotation period)'''

    print 'Period = ', p
    dur = (max(t) - min(t))

    # (crude estimate of) total number of spots needed during entire
    # time-series
    nspot_tot = int(nspot * dur / 2 / tau)

    # uniform distribution of spot longitudes
    lon = scipy.rand(nspot_tot) * 2 * scipy.pi

   # distribution of spot latitudes uniform in sin(latitude)
    lat = scipy.arcsin(scipy.rand(nspot_tot))

    # spot rotation rate optionally depends on latitude
    period = ((scipy.sin(lat) - 0.5) * diffrot + 1.0 ) * p
    period0 = scipy.ones(nspot_tot) * p

    # all spots have the same maximum area
    # (crude estimate of) filling factor needed per spot
    ff = amp / scipy.sqrt(nspot)
    scale_fac = 1
    amax = scipy.ones(nspot_tot) * ff * scale_fac

    # all spots have the evolution timescale
    decay = scipy.ones(nspot_tot) * tau

    # uniform distribution of spot peak times
    # start well before and end well after time-series limits (to
    # avoid edge effects)
    extra = 3 * decay.max()
    pk = scipy.rand(nspot_tot) * (dur + 2 * extra) - extra

    # COMPUTE THE LIGHT CURVE
    print "Computing light curve..."
    time = numpy.array(t - min(t))

    area_tot = scipy.zeros_like(time)
    dF_tot = scipy.zeros_like(time)
    dF_tot0 = scipy.zeros_like(time)

    # add up the contributions of individual spots
    for i in range(nspot_tot):

        # Spot area
        if (pk[i] == 0) + (decay[i] == 0):
            area = scipy.ones_like(time) * amax[i]
        else:
            area = amax[i] * \
                scipy.exp(-(time - pk[i])**2 / 2. / decay[i]**2)
        area_tot += area

        # Fore-shortening
        phase = 2 * scipy.pi * time / period[i] + lon[i]
        phase0 = 2 * scipy.pi * time / period0[i] + lon[i]
        mu = scipy.cos(incl) * scipy.sin(lat[i]) + \
            scipy.sin(incl) * scipy.cos(lat[i]) * scipy.cos(phase)
        mu0 = scipy.cos(incl) * scipy.sin(lat[i]) + \
            scipy.sin(incl) * scipy.cos(lat[i]) * scipy.cos(phase0)
        mu[mu < 0] = 0.0
        mu0[mu0 < 0] = 0.0

        # Flux
        dF_tot -= area * mu
        dF_tot0 -= area * mu0

    amp_eff = dF_tot.max()-dF_tot.min()
    nspot_eff = area_tot / scale_fac / ff

    res0 = scipy.array([nspot_eff.mean(), ff, amp_eff])
    res1 = scipy.zeros((4, len(time)))

    res1[0,:] = time
    res1[1,:] = area_tot
    res1[2,:] = dF_tot
    res1[3,:] = dF_tot0

    print 'Used %d spots in total over %d rotation periods.' % (nspot_tot, dur)
    print 'Mean filling factor of individual spots was %.4f.' % ff
    print 'Desired amplitude was %.4f, actual amplitude was %.4f.' \
            % (amp, amp_eff)
    print 'Desired number of spots at any one time was %d.' % nspot
    return res0, res1
