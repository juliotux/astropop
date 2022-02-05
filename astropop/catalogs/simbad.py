# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in Vizier catalogs."""

from functools import partial
import numpy as np
from multiprocessing.pool import Pool
from astroquery.simbad import Simbad, SimbadClass
from astropy.coordinates import SkyCoord
from astropy import units as u

from .base_catalog import _BasePhotometryCatalog
from ._online_tools import _timeout_retry, \
                           MAX_PARALLEL_QUERY, \
                           astroquery_radius, astroquery_skycoord
from ..py_utils import string_fix


__all__ = ['simbad_query_id', 'SimbadCatalog', 'SimbadCatalogClass']


def simbad_query_id(ra, dec, limit_angle, name_order=None,
                    simbad=None):
    """Query name ids for a star in Simbad.

    Parameters
    ----------
    ra, dec : `float`
        RA and DEC decimal degrees coordinates to query.
    limit_angle : string, float, `~astropy.coordinates.Angle`
        Maximum radius for search.
    name_order : `list`, optional
        Order of priority of name prefixes to query.
        Default: None
    simbad : `~astroquery.simbad.Simbad`, optional
        `~astroquery.simbad.Simbad` to be used in query.

    Returns
    -------
    `str`
        The ID of the object.
    """
    if name_order is None:
        name_order = ['MAIN_ID', 'NAME', 'HD', 'HR', 'HYP', 'TYC', 'AAVSO']

    if simbad is not None:
        s = simbad
    else:
        s = Simbad()

    def _strip_spaces(name):
        name = name.strip('NAME')
        name = name.strip('* ')
        # remove excessive spaces
        while '  ' in name:
            name = name.replace('  ', ' ')
        return name.strip(' ')

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
    return None


class SimbadCatalogClass(_BasePhotometryCatalog):
    """Base class to handle with Simbad."""

    id_key = 'MAIN_ID'
    ra_key = 'RA'
    dec_key = 'DEC'
    flux_key = 'FLUX_{band}'
    flux_error_key = 'FLUX_ERROR_{band}'
    flux_unit_key = 'FLUX_UNIT_{band}'
    flux_bibcode_key = 'FLUX_BIBCODE_{band}'
    type = 'online'
    prepend_id_key = False
    available_filters = ["U", "B", "V", "R", "I", "J", "H", "K", "u", "g", "r",
                         "i", "z"]
    _last_query_info = None
    _last_query_table = None
    _simbad = None

    @property
    def simbad(self):
        """Query operator instance."""
        if self._simbad is None:
            self._simbad = Simbad()
        self._simbad.ROW_LIMIT = 0
        return self._simbad()

    @simbad.setter
    def simbad(self, value):
        if not isinstance(value, SimbadClass):
            raise ValueError(f"{value} is not a SimbadClass instance.")
        self._simbad = value

    def get_simbad(self, band=None):
        """Get a copy of the simbad querier with optional band fields."""
        s = self.simbad
        if band is not None:
            s.add_votable_fields(f'fluxdata({band})')
        return s

    def _flux_keys(self, band):
        flux_key = self.flux_key.format(band=band)
        flux_error_key = self.flux_error_key.format(band=band)
        flux_bibcode_key = self.flux_bibcode_key.format(band=band)
        flux_unit_key = self.flux_unit_key.format(band=band)
        return flux_key, flux_error_key, flux_unit_key, flux_bibcode_key

    def query_object(self, center, band=None, **kwargs):
        """Query a single object in the catalog."""
        s = self.get_simbad(band)
        if not isinstance(center, str):
            raise ValueError("Simbad query_object only accept object name. "
                             "Try query_region instead.")
        return self._query(s.query_object, center, **kwargs)

    def query_region(self, center, radius, band=None, **kwargs):
        """Query all objects in a region."""
        s = self.get_simbad(band)
        center = astroquery_skycoord(center)
        radius = astroquery_radius(radius)
        return self._query(s.query_region, center, radius=radius, **kwargs)

    def _id_resolve(self, idn):
        if self.prepend_id_key:
            idn = [f"{self.id_key} {i}" for i in idn]
            idn = np.array(idn)
        return np.array([string_fix(i) for i in idn])

    def filter_flux(self, band, query=None):
        """Filter the flux data of a query."""
        self.check_filter(band)
        flux_key, error_key, unit_key, bibcode_key = self._flux_keys(band)
        if query is None:
            query = self._last_query_table

        if flux_key not in query.colnames:
            raise KeyError(f'Simbad query must be performed with band {band}'
                           ' for flux data.')

        flux = np.array(query[flux_key].data)
        flux_error = np.array(query[error_key].data)
        bibcode = np.array(query[bibcode_key].data)
        unit = np.array(query[unit_key].data)

        return flux, flux_error, unit, bibcode

    def match_objects(self, ra, dec, band=None, limit_angle='2 arcsec'):
        """Match objects from RA DEC list with this catalog."""
        flux_keys = ['flux', 'flux_error', 'flux_unit', 'flux_bibcode']
        table_props = [('id', ''), ('ra', np.nan), ('dec', np.nan),
                       ('flux', np.nan), ('flux_error', np.nan),
                       ('flux_unit', ''), 'flux_bibcode', '']
        res = self._match_objects(ra, dec, band, limit_angle,
                                  flux_keys, table_props)

        return res

    @staticmethod
    def match_object_ids(ra, dec, limit_angle='2 arcsec', name_order=None):
        """Get the id from Simbad for every object in a RA, Dec list."""
        # Perform it in parallel to handle the online query overhead
        func = partial(simbad_query_id, name_order=name_order,
                       limit_angle=limit_angle)
        p = Pool(MAX_PARALLEL_QUERY)
        results = p.starmap(func, list(zip(ra, dec)))
        return results


SimbadCatalog = SimbadCatalogClass()
