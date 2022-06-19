# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in Vizier catalogs."""

import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astropy import units as u

from ._sources_catalog import _SourceCatalogClass
from ._online_tools import _timeout_retry, astroquery_query
from ..py_utils import string_fix


__all__ = ['simbad_query_id', 'SimbadSourcesCatalog']


def _simbad_query_id(ra, dec, limit_angle, name_order=None):
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


simbad_query_id = np.vectorize(_simbad_query_id, excluded=['limit_angle', 'name_order'])


class SimbadSourcesCatalog(_SourceCatalogClass):
    """Sources catalog from Simbad plataform."""

    _available_filters = ['B', 'V', 'R', 'I', 'J', 'H', 'K',
                          'u', 'g', 'r', 'i', 'z']

    def _setup_catalog(self):
        self._s = Simbad()
        self._s.add_votable_fields('pm')
        self._s.ROW_LIMIT = 0
        if self._band is not None:
            self._s.add_votable_fields(f'fluxdata({self._band})')

    def _do_query(self):
        # We query everything in J2000 and query propermotion too
        self._query = astroquery_query(self._s.query_region,
                                       self._center,
                                       radius=self._radius,
                                       epoch='J2000')
        ids = np.array([string_fix(i) for i in self._query['MAIN_ID']])
        if self._band is not None:
            mags = self._query[f'FLUX_{self._band}']
            mags_error = self._query[f'FLUX_ERROR_{self._band}']
        else:
            mags = mags_error = None

        self._set_values(ids=ids,
                         ra=self._query['RA'], dec=self._query['DEC'],
                         mag=mags, mag_error=mags_error,
                         pm_ra=self._query['PMRA'],
                         pm_dec=self._query['PMDEC'],
                         frame='icrs', obstime='J2000',
                         radec_unit=('hourangle', 'degree'))
