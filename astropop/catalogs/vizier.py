# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in Vizier catalogs."""

import abc
import numpy as np
import yaml
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astropy.time import Time

from ._sources_catalog import _OnlineSourcesCatalog, SourcesCatalog
from ._online_tools import astroquery_query
from ..py_utils import string_fix
from ..math import qfloat


__all__ = ['_VizierSourcesCatalog']


class _VizierSourcesCatalog(_OnlineSourcesCatalog, abc.ABC):
    """Sources catalog from Vizier plataform.

    Parameters
    ----------
    config_file: string
        Yaml configuration file containing the parameters to be used
        for catalog queries, like table id, column names, etc.
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
        list will be queried.

    Raises
    ------
    ValueError:
        If a ``band`` not available in the filters is passed.
    """

    def __init__(self, config_file, *args, **kwargs):
        self._conf = yaml.safe_load(config_file)
        self._setup_vizier()
        super().__init__(*args, **kwargs)

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


class DENISSourcesCatalog(_VizierSourcesCatalog):
    _table = 'B/denis'


class TwoMASSSourcesCatalog(_VizierSourcesCatalog):
    _table = 'II/246/out'


class VSXSourcesCatalog(_VizierSourcesCatalog):
    _table = 'B/vsx'


class GCVSSourcesCatalog(_VizierSourcesCatalog):
    _table = 'B/gcvs'


class AllWISESourcesCatalog(_VizierSourcesCatalog):
    _table = 'II/328/allwise'


class WISESourcesCatalog(_VizierSourcesCatalog):
    _table = 'II/311/wise'


class UnWISESourcesCatalog(_VizierSourcesCatalog):
    _table = 'II/363/unwise'


class CatWISE2020SourcesCatalog(_VizierSourcesCatalog):
    _table = 'cat/II/365'


class HipparcosSourcesCatalog(_VizierSourcesCatalog):
    _table = 'I/239/hip_main'


class TychoSourcesCatalog(_VizierSourcesCatalog):
    _table = 'I/239/tyc_main'
