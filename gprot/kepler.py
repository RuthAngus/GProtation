from __future__ import print_function, division

import re
import numpy as np
import pandas as pd
from time import sleep
import logging

from collections import OrderedDict

import kplr
from kplr.api import APIError


from .lc import LightCurve, qtr_times
from .model import GPRotModel

client = None
offline_client = None

class KeplerGPRotModel(GPRotModel):
    """Parameters are A, l, G, sigma, period

    Bounds and priors are adjusted based on population results
    with the default settings.
    """
    _default_bounds = ((-20., 0.), 
               (2, 8.), 
               (0., 3.), 
               (-20., 0.), 
               (-0.69, 4.61)) 

    _default_gp_prior_mu = (-13, 5.0, 1.9, -17)
    _default_gp_prior_sigma = (5.7, 1.2, 1.4, 5)


class KeplerLightCurve(LightCurve):
    """
    Light curve from Kepler data

    Parameters
    ----------
    kid : int
        Either KOI number or KIC number.

    sub : int
        Subsampling factor

    nsigma : float
        Threshold for sigma-clipping

    chunksize : int
        (Approximate) number of points in each subchunk of the light curve.
    """
    def __init__(self, kid, sub=1, nsigma=5, chunksize=200,
                 quarters=None, normalized=True, careful_stitching=False,
                 sap=False, offline=False):

        if kid < 10000:
            self.koinum = int(kid)
            self._kepid = None
        else:
            self.koinum = None
            self._kepid = kid

        if quarters is None:
            self.quarters = None
        else:
            try:
                self.quarters = sorted(quarters)
            except TypeError:
                self.quarters = [quarters]

        self.offline = offline

        self.sub = sub
        self.nsigma = nsigma
        self.chunksize = chunksize
        self.normalized = normalized
        self.sap = sap        
        self.careful_stitching = careful_stitching

        self._x = None
        self._y = None
        self._yerr = None

        self._x_list = None
        self._y_list = None
        self._yerr_list = None

        self._x_full = None
        self._y_full = None
        self._yerr_full = None

    @property
    def client(self):
        global offline_client
        global client
        if self.offline:
            if offline_client is None:
                offline_client = kplr.OfflineAPI()
            return offline_client
        else:
            if client is None:
                client = kplr.API()
            return client

    @property
    def is_koi(self):
        return self.koinum is not None

    @property
    def name(self):
        if not hasattr(self, '_name') or self._name is None:
            if self.is_koi:
                name = 'KOI-{}'.format(self.koinum)
            else:
                name = 'KIC-{}'.format(self.kepid)

            if self.quarters is not None:
                for q in self.quarters:
                    name += '-Q{}'.format(q)
            self._name = name

        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def kepid(self):
        if self._kepid is None:
            koi = self.client.koi(self.koinum + 0.01)
            self._kepid = koi.kepid
        return self._kepid            

    def _get_data(self, clobber=False):
        # Query kplr API to get KOI/KIC info.
        if self.is_koi:
            star = self.client.koi(self.koinum + 0.01)                
            kois = [self.client.koi(self.koinum + 0.01*i) for i in range(1, star.koi_count+1)]
        else:
            star = self.client.star(self.kepid)
            kois = []

        # Get a list of light curve datasets.
        lcs = star.get_light_curves(short_cadence=False, clobber=clobber)

        # Loop over the datasets and read in the data.
        time, flux, ferr = [], [], []
        p_last = None
        t_last = None
        for lc in lcs:
            with lc.open() as f:
                # The lightcurve data are in the first FITS HDU.
                hdu_data = f[1].data
                t = hdu_data["time"]

                if self.sap:
                    f = hdu_data['sap_flux']
                    f_e = hdu_data['sap_flux_err']
                else:
                    f = hdu_data["pdcsap_flux"]
                    f_e = hdu_data["pdcsap_flux_err"]
                q = hdu_data["sap_quality"]

                # Keep only good points, median-normalize and mean-subtract flux
                m = np.logical_not(q) & np.isfinite(f) & np.isfinite(f_e)

                if self.quarters is not None:
                    ok = False
                    for qtr in self.quarters:
                        t0, t1 = qtr_times.ix[qtr, ['tstart', 'tstop']]
                        # print(qtr, t0, t1, t[m].min(), t[m].max())
                        if (np.absolute(t0 - t[m].min()) < 10 and
                            np.absolute(t1 - t[m].max()) < 10):
                            ok = True
                    if not ok:
                        continue

                time.append(t[m])
                if self.normalized:
                    norm = np.median(f[m])
                    flux.append(f[m] / norm - 1)
                    ferr.append(f_e[m] / norm)
                else:
                    flux.append(f[m])
                    ferr.append(f_e[m])

                if self.careful_stitching:
                    if not self.sap:
                        raise NotImplementedError('Do not use "careful_stitching" option if not using SAP data.')
                    # Use polynomial fit from last quarter to set level.
                    if p_last is not None:
                        t0 = time[-1][0]

                        # If a quarter is missing, don't try to fit polynomial.
                        if t0 - t_last > 20:
                            f_initial = flux[-2][-1]
                        else:
                            f_initial = np.polyval(p_last, t0)
                        f_offset = f_initial - flux[-1][0]
                        flux[-1] += f_offset

                    # Fit 3rd-degree polynomial to tail of quarter to 
                    # set initial level for next quarter
                    p_last = np.polyfit(time[-1], flux[-1], 3)
                    t_last = time[-1][-1]

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

        # Mask times > 1581 to avoid penultimate safe mode
        # Also, require to be at least > Q2
        if self.quarters is None:
            m |= time > 1581
            m |= time < 260

        self._x = time[~m]
        self._y = flux[~m]
        self._yerr = ferr[~m]

        self._x_full = time[~m].copy()
        self._y_full = flux[~m].copy()
        self._yerr_full = ferr[~m].copy()

        self.sigma_clip(self.nsigma)
        self.subsample(self.sub)

    # def multi_split_quarters(self):
    #     if self.quarters is None:
    #         qtrs = qtr_times.index
    #     else:
    #         qtrs = self.quarters

    #     N = len(qtrs)
    #     subs = np.ones(len(qtrs))*40
    #     # have middle be 5, flanked by 10, 20, then 40
    #     for i, sub in zip(range(3), [5,10,20]):
    #         subs[N//2 + i] = sub
    #         subs[N//2 - i] = sub

    #     super(KeplerLightCurve, self).multi_split_quarters(qtrs, subs, seed=self.kepid)

    def subsample(self, *args, **kwargs):
        if 'seed' not in kwargs:
            kwargs['seed'] = self.kepid
        super(KeplerLightCurve, self).subsample(*args, **kwargs)

    def make_best_chunks(self, *args, **kwargs):
        if 'seed' not in kwargs:
            kwargs['seed'] = self.kepid
        super(KeplerLightCurve, self).make_best_chunks(*args, **kwargs)

    # def _make_chunks(self, *args, **kwargs):
    #     self._split_quarters()

