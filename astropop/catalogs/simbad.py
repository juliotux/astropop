# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in Simbad database."""

import numpy as np
import warnings
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.simbad import Simbad
from astropy import units as u

from ._sources_catalog import _OnlineSourcesCatalog, SourcesCatalog
from ._online_tools import _timeout_retry, astroquery_query
from ..py_utils import string_fix


__all__ = ['simbad_query_id', 'SimbadSourcesCatalog', 'simbad']


def _simbad_query_id(ra, dec, limit_angle, name_order=None):
    """Query name ids for a star in Simbad. See simbad_query_id.

    Parameters
    ----------
    ra, dec : `float`
        RA and DEC decimal degrees coordinates to query.
    limit_angle : string, float, `~astropy.coordinates.Angle`
        Maximum radius for search. If a string value is passed, it must be
        readable by astropy.coordinates.Angle. If a float value is passed,
        it will be interpreted as a decimal degree radius.
    name_order : `list`, optional
        Order of priority of name prefixes to query. None will use the default
        order of ['MAIN_ID', 'NAME', 'HD', 'HR', 'HYP', 'TYC', 'AAVSO'].
        Default: None

    Returns
    -------
    `str` or list(`str`)
        The ID of each object queried. If a single object is queried,
        a string is returned. If a list of objects is queried, a list
        of strings is returned.
    """
    if name_order is None:
        name_order = ['MAIN_ID', 'NAME', 'HD', 'HR', 'HYP', 'TYC', 'AAVSO']

    name_order = np.atleast_1d(name_order)

    s = Simbad()

    def _strip_spaces(name):
        name = name.strip('NAME')
        name = name.strip('* ')
        # remove excessive spaces
        while '  ' in name:
            name = name.replace('  ', ' ')
        return name.strip(' ')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        q = _timeout_retry(s.query_region, SkyCoord(ra, dec,
                                                    unit=(u.degree, u.degree)),
                           radius=limit_angle)

    if q is not None:
        name = string_fix(q['MAIN_ID'][0])
        ids = _timeout_retry(s.query_objectids, name)['ID']
        for i in name_order:
            if i == 'MAIN_ID':
                return _strip_spaces(name)
            for k in ids:
                if i+' ' in k:
                    return _strip_spaces(k)
    # If nothing is found, return empty string
    return ''


def simbad_query_id(ra, dec, limit_angle, name_order=None):
    """Query name ids for a star in Simbad.

    Parameters
    ----------
    ra, dec : `float` or list(`float`)
        RA and DEC decimal degrees coordinates to query. If more than one
        object is queried, ra and dec must be lists of the same length.
    limit_angle : string, float, `~astropy.coordinates.Angle`
        Maximum radius for search.
    name_order : `list`, optional
        Order of priority of name prefixes to query. None will use the default
        order of ['MAIN_ID', 'NAME', 'HD', 'HR', 'HYP', 'TYC', 'AAVSO'].
        Default: None

    Returns
    -------
    `str` or list(`str`)
        The ID of each object queried. If a single object is queried,
        a string is returned. If a list of objects is queried, a list
        of strings is returned.
    """
    f = np.vectorize(_simbad_query_id, excluded=['limit_angle', 'name_order'])
    query = f(ra, dec, limit_angle, name_order=name_order)
    if query.size == 1:
        return str(query)
    return list(query)


class SimbadSourcesCatalog(_OnlineSourcesCatalog):
    """Sources catalog from Simbad plataform.

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
        Filters to query photometric informations. If `None`, photometric
        informations will be disabled. If ``'all'`` (default), all
        available filters will be queried. If a list, all filters in that
        list will be queried. By default, `None`.

    Raises
    ------
    ValueError:
        If a ``band`` not available in the filters is passed.
    """

    _available_filters = ['B', 'V', 'R', 'I', 'J', 'H', 'K',
                          'u', 'g', 'r', 'i', 'z']

    def __init__(self, center, radius, band=None):
        # Just change the default behavior of band to None
        super(SimbadSourcesCatalog, self).__init__(center, radius, band)

    def coordinates_bibcode(self):
        return np.array(self._query['COO_BIBCODE'])

    def magnitudes_bibcode(self, band):
        self._ensure_band(band)
        return np.array(self._query[f'FLUX_BIBCODE_{band}'])

    def _setup_catalog(self):
        self._s = Simbad()
        self._s.add_votable_fields('pm')
        self._s.ROW_LIMIT = 0
        for filt in self.filters:
            self._s.add_votable_fields(f'fluxdata({filt})')

    def _do_query(self):
        # We query everything in J2000 and query propermotion too
        self._query = astroquery_query(self._s.query_region,
                                       self._center,
                                       radius=self._radius,
                                       epoch='J2000')
        ids = np.array([string_fix(i) for i in self._query['MAIN_ID']])
        for band in self.filters:
            if self._mags_table is None:
                self._mags_table = Table()
                self._mags_bib = {}
            m = np.array(self._query[f'FLUX_{band}'])
            mags_error = np.array(self._query[f'FLUX_ERROR_{band}'])
            self._mags_table[f'{band}'] = m
            self._mags_table[f'{band}_error'] = mags_error

        sk = SkyCoord(self._query['RA'], self._query['DEC'],
                      unit=('hourangle', 'degree'),
                      pm_ra_cosdec=self._query['PMRA'],
                      pm_dec=self._query['PMDEC'],
                      obstime='J2000.0', frame='icrs')

        SourcesCatalog.__init__(self, sk, ids=ids)


simbad = SimbadSourcesCatalog
