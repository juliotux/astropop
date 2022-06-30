# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in catalogs using TAP services."""

import copy
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.time import Time

from ._sources_catalog import _OnlineSourcesCatalog, SourcesCatalog
from ._online_tools import astroquery_query, string_fix


__all__ = ['GaiaDR3SourcesCatalog']


class GaiaDR3SourcesCatalog(_OnlineSourcesCatalog):
    _available_filters = ['G', 'BP', 'RP']
    _columns = ['DESIGNATION', 'ref_epoch', 'ra', 'dec', 'pmra', 'pmdec',
                'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                'phot_g_mean_flux_over_error', 'phot_bp_mean_flux_over_error',
                'phot_rp_mean_flux_over_error']

    def _setup_catalog(self):
        self._g = copy.deepcopy(Gaia)
        self._g.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
        self._g.ROW_LIMIT = -1

    def _do_query(self):
        self._query = astroquery_query(self._g.cone_search,
                                       self._center,
                                       radius=self._radius,
                                       columns=self._columns).results
        sk = SkyCoord(self._query['ra'], self._query['dec'],
                      obstime=Time(self._query['ref_epoch'], format='jyear'))
        ids = np.array([string_fix(i) for i in self._query['designation']])

        if self._band is not None:
            band = self._band.lower()
            mag = np.array(self._query[f'phot_{band}_mean_mag'])
            # Magnitude errors must be computed from SNR
            # sigma(mag) approx 1.1/snr
            mag_er = 1.1/np.array(self._query[f'phot_{band}'
                                              '_mean_flux_over_error'])
        else:
            mag = None
            mag_er = None

        SourcesCatalog.__init__(self, sk, ids=ids, mag=mag, mag_error=mag_er,
                                mag_unit='mag')
