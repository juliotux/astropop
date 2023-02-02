# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in catalogs using TAP services."""

import copy
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.time import Time

from ._sources_catalog import _OnlineSourcesCatalog, SourcesCatalog
from ._online_tools import astroquery_query, string_fix
from ..math import QFloat


__all__ = ['GaiaDR3SourcesCatalog', 'gaiadr3']


class GaiaDR3SourcesCatalog(_OnlineSourcesCatalog):
    """Sources catalog from Gaia-DR3 catalog.

    This class just wraps around `~astroquery.gaia.Gaia` class.

    Parameters
    ----------
    center: string, tuple or `~astropy.coordinates.SkyCoord`
        The center of the search field.
        If center is a string, can be an object name or the string
        containing the object coordinates. If it is a tuple, have to be
        (ra, dec) coordinates, in hexa or decimal degrees format.
    radius: string, float, `~astropy.coordinates.Angle` (optional)
        The radius to search. If None, the query will be performed as
        single object query mode. Else, the query will be performed as
        field mode. If a string value is passed, it must be readable by
        astropy.coordinates.Angle. If a float value is passed, it will
        be interpreted as a decimal degree radius.
    band: string or list(string) (optional)
        Filters to query photometric informations. If None, photometric
        informations will be disabled. If ``'all'`` (default), all
        available filters will be queried. If a list, all filters in that
        list will be queried. By default, all filters are available.

    Raises
    ------
    ValueError:
        If a ``band`` not available in the filters is passed.
    """

    _available_filters = ['G', 'BP', 'RP']
    _columns = ['DESIGNATION', 'ref_epoch', 'ra', 'dec', 'pmra', 'pmdec',
                'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                'phot_g_mean_flux_over_error', 'phot_bp_mean_flux_over_error',
                'phot_rp_mean_flux_over_error']

    def _setup_catalog(self):
        self._g = copy.deepcopy(Gaia)
        self._g.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
        self._g.ROW_LIMIT = -1

    @staticmethod
    def _filter_magnitudes(query, band):
        """Get the qfloat magnitudes."""
        f = band.lower()
        mag = np.array(query[f'phot_{f}_mean_mag'])
        # Magnitude errors must be computed from SNR
        # sigma(mag) approx 1.1/snr
        mag_err = 1.1/np.array(query[f'phot_{f}_mean_flux_over_error'])
        return QFloat(mag, mag_err, unit='mag')

    def _do_query(self):
        self._query = astroquery_query(self._g.cone_search,
                                       self._center,
                                       radius=self._radius,
                                       columns=self._columns).results
        sk = SkyCoord(self._query['ra'], self._query['dec'],
                      obstime=Time(self._query['ref_epoch'], format='jyear'))
        ids = np.array([string_fix(i) for i in self._query['DESIGNATION']])

        # perform magnitude filtering only if available
        mag = {}
        for f in self.filters:
            mag[f] = self._filter_magnitudes(self._query, f)

        SourcesCatalog.__init__(self, sk, ids=ids, mag=mag)


gaiadr3 = GaiaDR3SourcesCatalog
