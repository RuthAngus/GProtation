from __future__ import print_function, division

import numpy as np
import pandas as pd

import kplr

from .lc import LightCurve

client = None

class KeplerLightCurve(LightCurve):
    def __init__(self, kid, sub=1, nsigma=5, chunksize=200):

        if koinum < 10000:
            self.koinum = int(kid)
            self._kepid = None
        else:
            self.koinum = None
            self._kepid = kid

        self.sub = sub
        self.nsigma = nsigma
        self.chunksize = chunksize

        self._x = None
        self._y = None
        self._yerr = None

        self._x_list = None
        self._y_list = None
        self._yerr_list = None

    @property
    def is_koi(self):
        return self.koinum is not None

    @property
    def name(self):
        if self.is_koi:
            return 'KOI-{}'.format(self.koinum)
        else:
            return 'KIC-{}'.format(self.kepid)

    @property
    def kepid(self):
        if self._kepid is None:
            global client
            if client is None:
                client = kplr.API()
            koi = client.koi(self.koinum + 0.01)
            self._kepid = koi.kepid
        return self._kepid            

    def _get_data(self, clobber=False):
        global client
        if client is None:
            client = kplr.API()

        if self.is_koi:
            star = client.koi(self.koinum + 0.01)
            kois = [client.koi(self.koinum + 0.01*i) for i in range(1, koi.koi_count+1)]
        else:
            star = client.star(self.kepid)
            kois = []

        # Get a list of light curve datasets.
        lcs = star.get_light_curves(short_cadence=False, clobber=clobber)

        # Loop over the datasets and read in the data.
        time, flux, ferr = [], [], []
        for lc in lcs:
            with lc.open() as f:
                # The lightcurve data are in the first FITS HDU.
                hdu_data = f[1].data
                t = hdu_data["time"]
                f = hdu_data["sap_flux"]
                f_e = hdu_data["sap_flux_err"]
                q = hdu_data["sap_quality"]

                # Keep only good points, median-normalize and mean-subtract flux
                m = np.logical_not(q)

                time.append(t[m])
                norm = np.median(f[m])
                flux.append(f[m] / norm - 1)
                ferr.append(f_e[m] / norm)

        time = np.concatenate(time)
        flux = np.concatenate(flux)
        ferr = np.concatenate(ferr)

        # Mask transits for all kois 
        m = np.zeros(len(time), dtype=bool)
        for k in kois:
            period, epoch = k.koi_period, k.koi_time0bk
            phase = (time + period/2. - epoch) % period - (period/2)
            duration = k.koi_duration / 24.
            m |= np.absolute(phase) < duration*0.55

        self._x = time[~m]
        self._y = flux[~m]
        self._yerr = ferr[~m]

        self.sigma_clip(self.nsigma)
        self.subsample(self.sub)

    def subsample(self, *args, **kwargs):
        if 'seed' not in kwargs:
            kwargs['seed'] = self.koinum
        super(KeplerLightCurve, self).subsample(*args, **kwargs)

    @property
    def x(self):
        if self._x is None:
            self._get_data()
        return self._x

    @x.setter
    def x(self, val):
        self._x = val

    @property
    def y(self):
        if self._y is None:
            self._get_data()
        return self._y

    @y.setter
    def y(self, val):
        self._y = val

    @property
    def yerr(self):
        if self._yerr is None:
            self._get_data()
        return self._yerr

    @yerr.setter
    def yerr(self, val):
        self._yerr = val
