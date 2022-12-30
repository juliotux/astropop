# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in Vizier catalogs."""

import abc
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astropy.time import Time

from ._sources_catalog import _OnlineSourcesCatalog, SourcesCatalog
from ._online_tools import astroquery_query
from ..py_utils import string_fix
from ..math import qfloat


__all__ = ['UCAC4SourcesCatalog', 'APASS9SourcesCatalog']


class _VizierSourcesCatalog(_OnlineSourcesCatalog, abc.ABC):
    """Sources catalog from Vizier plataform."""

    _table = None
    _frame = 'icrs'
    _columns = ['+_r', '**']

    @staticmethod
    @abc.abstractmethod
    def _filter_magnitudes(query, band):
        """Get the qfloat magnitudes."""

    @staticmethod
    @abc.abstractmethod
    def _filter_coordinates(query, obstime, frame):
        """Get the SkyCoord coordinates."""

    @staticmethod
    @abc.abstractmethod
    def _filter_ids(query):
        """Get the id names for the objects."""

    @staticmethod
    @abc.abstractmethod
    def _filter_epoch(query):
        """Get the epoch for the coordinates."""

    def _setup_catalog(self):
        self._v = Vizier(catalog=self._table, columns=self._columns)
        self._v.ROW_LIMIT = -1

    def _do_query(self):
        self._query = astroquery_query(self._v.query_region,
                                       self._center,
                                       radius=self._radius)[0]
        ids = self._filter_ids(self._query)

        # perform magnitude filtering only if available
        mag = {}
        for filt in self.filters:
            mag[filt] = self._filter_magnitudes(self._query, filt)

        obstime = self._filter_epoch(self._query)
        sk = self._filter_coordinates(self._query, obstime, self._frame)

        SourcesCatalog.__init__(self, sk, ids=ids, mag=mag)


def _ucac4_filter_coord(query, obstime, frame,
                        rakey='RAJ2000', deckey='DEJ2000',
                        pmrakey='pmRA', pmdeckey='pmDE'):
    ra = np.array(query[rakey])*query[rakey].unit
    dec = np.array(query[deckey])*query[deckey].unit
    pmra = np.array(query[pmrakey])*query[pmrakey].unit
    pmdec = np.array(query[pmdeckey])*query[pmdeckey].unit
    return SkyCoord(ra, dec, frame=frame, obstime=obstime,
                    pm_ra_cosdec=pmra, pm_dec=pmdec)


def _ucac4_filter_magnitude(query, band):
    unit = query[f'{band}mag'].unit
    mag = np.array(query[f'{band}mag'])
    if f'e_{band}mag' in query.colnames:
        err_unit = query[f'e_{band}mag'].unit
        mag_err = np.array(query[f'e_{band}mag'])
        mag_err = np.array([float(i) if i != '' else np.nan
                            for i in mag_err])
        if str(err_unit) == 'cmag':
            mag_err /= 100.0
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


class APASS9SourcesCatalog(_VizierSourcesCatalog):
    _table = 'apass9'
    _available_filters = ['V', 'B', "g", "r", "i"]
    _columns = ['+_r', '**']

    @staticmethod
    def _filter_coordinates(query, obstime, frame):
        ra = np.array(query['RAJ2000'])*query['RAJ2000'].unit
        dec = np.array(query['DEJ2000'])*query['DEJ2000'].unit
        return SkyCoord(ra, dec, frame=frame, obstime=obstime)

    @staticmethod
    def _filter_ids(query):
        return ['']*len(query)

    @staticmethod
    def _filter_epoch(query):
        return None

    @staticmethod
    def _filter_magnitudes(query, band):
        if band in ['g', 'r', 'i']:
            band = f'{band}_'
        return _ucac4_filter_magnitude(query, band)


class GSC242SourcesCatalog(_VizierSourcesCatalog):
    _table = 'I/353/gsc242'
    _available_filters = ['G', 'Bj', 'Fpg', 'Epg', 'Npg',
                          'U', 'B', 'V', 'u', 'g', 'r', 'i', 'z',
                          'y', 'J', 'H', 'Ks', 'Z', 'Y', 'W1', 'W2',
                          'W3', 'W4', 'FUV', 'NUV', 'RP', 'BP']
    _filter_magnitudes = staticmethod(_ucac4_filter_magnitude)

    @property
    def _columns(self):
        cols = ['+_r', 'GSC2', 'RA_ICRS', 'DE_ICRS', 'pmRA', 'pmDE', 'Epoch']
        for i in self.filters:
            cols += [f'{i}mag', f'e_{i}mag']
        return cols

    @staticmethod
    def _filter_epoch(query):
        key = 'Epoch'
        if key not in query.colnames:
            key = '_tab1_11'
        return Time(query[key], format='jyear')

    @staticmethod
    def _filter_ids(query):
        return [f'GSC2 {string_fix(i)}' for i in list(query['GSC2'])]

    @staticmethod
    def _filter_coordinates(query, obstime, frame):
        return _ucac4_filter_coord(query, obstime, frame,
                                   rakey='RA_ICRS', deckey='DE_ICRS')


class UCAC5SourcesCatalog(_VizierSourcesCatalog):
    _table = 'I/340'
    _available_filters = ['G', 'R', 'J', 'H', 'K', 'f.']
    _filter_magnitudes = staticmethod(_ucac4_filter_magnitude)

    @staticmethod
    def _filter_epoch(query):
        # Gaia epoch of coordinates
        return Time('J2015.0', format='jyear')

    @staticmethod
    def _filter_ids(query):
        return [f'Gaia {string_fix(i)}' for i in list(query['SrcIDgaia'])]

    @staticmethod
    def _filter_coordinates(query, obstime, frame):
        return _ucac4_filter_coord(query, obstime, frame,
                                   rakey='RAgaia', deckey='DEgaia')

# TODO:
# - DENIS: B/denis
# - 2MASS
# - VSX: B/vsx
# - GCVS: B/gcvs
