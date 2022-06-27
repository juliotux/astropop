# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in Vizier catalogs."""

import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astropy.time import Time

from ._sources_catalog import _OnlineSourcesCatalog, SourcesCatalog
from ._online_tools import astroquery_query
from ..py_utils import string_fix
from ..math import qfloat


__all__ = ['UCAC4SourcesCatalog']


class _VizierSourcesCatalog(_OnlineSourcesCatalog):
    """Sources catalog from Vizier plataform."""

    _filter_magnitudes = None
    _filter_coordinates = None
    _filter_ids = None
    _filter_epoch = None
    _table = None
    _frame = 'icrs'
    _columns = ['*']

    def _setup_catalog(self):
        if self._filter_magnitudes is None or \
           self._filter_coordinates is None or \
           self._filter_epoch is None or \
           self._table is None:
            raise NotImplementedError('Some required methods are not '
                                      'properly setup.')
        self._v = Vizier(catalog=self._table, columns=self._columns)
        self._v.ROW_LIMIT = -1

    def _do_query(self):
        self._query = astroquery_query(self._v.query_region,
                                       self._center,
                                       radius=self._radius)[0]
        ids = self._filter_ids(self._query)
        if self._band is not None:
            mag = self._filter_magnitudes(self._query, self._band)
        else:
            mag = None
        obstime = self._filter_epoch(self._query)
        sk = self._filter_coordinates(self._query, obstime, self._frame)

        SourcesCatalog.__init__(self, sk, ids=ids, mag=mag)


def _ucac4_filter_coord(query, obstime, frame):
    ra = np.array(query['RAJ2000'])*query['RAJ2000'].unit
    dec = np.array(query['DEJ2000'])*query['DEJ2000'].unit
    pmra = np.array(query['pmRA'])*query['pmRA'].unit
    pmdec = np.array(query['pmDE'])*query['pmDE'].unit
    return SkyCoord(ra, dec, frame=frame, obstime=obstime,
                    pm_ra_cosdec=pmra, pm_dec=pmdec)


def _ucac4_filter_magnitude(query, band):
    unit = query[f'{band}mag'].unit
    mag = np.array(query[f'{band}mag'])
    if f'e_{band}mag' in query:
        mag_err = np.array(query[f'e_{band}mag'])
        mag_err = [float(i) if i != '' else np.nan
                   for i in mag_err]
    else:
        mag_err = None
    return qfloat(mag, uncertainty=mag_err, unit=unit)


class UCAC4SourcesCatalog(_VizierSourcesCatalog):
    _table = 'UCAC4'
    _filter_coordinates = staticmethod(_ucac4_filter_coord)
    _filter_magnitudes = staticmethod(_ucac4_filter_magnitude)
    _available_filters = ['J', 'H', 'K', 'B', 'V', 'g', 'r', 'i']

    @staticmethod
    def _filter_ids(query):
        return [f'UCAC4 {string_fix(i)}' for i in list(query['UCAC4'])]

    @staticmethod
    def _filter_epoch(query):
        return Time(query['EpRA'], format='jyear')

    @property
    def _columns(self):
        cols = ['+_r', 'UCAC4', 'RAJ2000', 'DEJ2000', 'pmRA', 'pmDE', 'EpRA']
        for i in self._available_filters:
            cols += [f'{i}mag', f'e_{i}mag']
        return cols
