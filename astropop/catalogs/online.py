# Licensed under a 3-clause BSD style license - see LICENSE.rst
import six
import copy
import numpy as np
from multiprocessing.pool import Pool
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery import ReadTimeout
from astropy.coordinates import SkyCoord
from astropy import units as u

from .base_catalog import _BasePhotometryCatalog, match_indexes
from ..logger import logger
from ..astrometry.coords_utils import guess_coordinates
from ..py_utils import string_fix


MAX_PARALLEL_QUERY = 30
MAX_RETRIES_TIMEOUT = 10


def _timeout_retry(func, *args, **kwargs):
    tried = kwargs.pop('_____retires', 0)
    try:
        q = func(*args, **kwargs)
    except ReadTimeout:
        if tried >= MAX_RETRIES_TIMEOUT:
            logger.warn('TimeOut obtained in 10 tries, aborting.')
            return
        return _timeout_retry(func, *args, **kwargs, _____retires=tried+1)
    return q


def get_center_radius(ra, dec):
    """Get a list of RA and DEC coordinates and returns the center and the
    search radius."""
    center_ra = (np.max(ra) + np.min(ra))/2
    center_dec = (np.max(dec) + np.min(dec))/2
    radius = np.max([np.max(ra) - np.min(ra),
                     np.max(dec) - np.min(dec)])
    return center_ra, center_dec, radius


def get_center_skycoord(center):
    if isinstance(center, six.stringtypes):
        return SkyCoord(center)
    elif isinstance(center, (tuple, list, np.ndarray)) \
         and len(center) == 2:
        return guess_coordinates(center[0], center[1])
    elif isinstance(center, SkyCoord):
        return center

    raise ValueError('Center coordinates {} not undertood.'
                     .format(center))



class _VizierCatalog(_BasePhotometryCatalog):
    """Base class to handle with Vizier online catalogs"""
    vizier_table = None
    id_key = None
    ra_key = 'RAJ2000'
    dec_key = 'DEJ2000'
    flux_key = None
    flux_error_key = None
    type = 'online'
    prepend_id_key = False

    def __init__(self):
        self.vizier = Vizier()
        self.vizier.ROW_LIMIT = -1

    def _flux_keys(self, filter):
        flux_key = self.flux_key.format(filter)
        if self.flux_error_key is not None:
            flux_error_key = self.flux_error_key.format(filter)
        else:
            flux_error_key = None
        return flux_key, flux_error_key

    def _get_center(self, center):
        return get_center_skycoord(center)

    def _query_vizier(self, center, radius, table):
        '''Performs the query in vizier site.'''
        # check if the new query is equal to previous one. Only perform a new
        # query if it is not redundant
        query_info = {'radius': radius, 'center': center, 'table': table}
        if query_info == self._last_query_info:
            logger.debug("Loading cached query.")
            return

        if table is None:
            raise ValueError("No Vizier table was defined.")

        logger.info("Performing Vizier query with: center:{} radius:{}"
                    " vizier_table:{}".format(center, radius, table))

        self.flush()
        self._last_query_info = query_info

        center = self._get_center(center)

        if radius is not None:
            radius = self._get_radius(radius)*u.degree,
            query = self.vizier.query_region(center, radius=radius,
                                             catalog=table)
        else:
            query = self.vizier.query_object(center, catalog=table)
        if len(query) == 0:
            raise RuntimeError("No online catalog results were found.")
        self._last_query_table = query[0]
        return copy.copy(self._last_query_table)

    def query_object(self, center):
        return self._query_vizier(center, radius=None, table=self.vizier_table)

    def query_region(self, center, radius):
        return self._query_vizier(center, radius, table=self.vizier_table)

    def query_ra_dec(self, center, radius):
        if self.ra_key is None or self.dec_key is None:
            raise ValueError("Invalid RA or Dec keys.")
        self._query_vizier(center, radius, table=self.vizier_table)
        ra = self._last_query_table[self.ra_key].data
        dec = self._last_query_table[self.dec_key].data

        # Solve the most common types of coordinates
        coords = [guess_coordinates(*c) for c in zip(ra, dec)]
        ra = np.array([c.ra.degree for c in coords])
        dec = np.array([c.dec.degree for c in coords])

        return ra, dec

    def query_id(self, center, radius):
        if self.id_key is None:
            raise ValueError("Invalid ID key.")
        self._query_vizier(center, radius, table=self.vizier_table)

        id = self._last_query_info[self.id_key].data
        if self.prepend_id_key:
            id = ["{id_key} {id}".format(id_key=self.id_key, id=i) for i in id]
            id = np.array(id)

        return id

    def query_flux(self, center, radius, filter):
        self.check_filter(filter)
        flux_key, flux_error_key = self._flux_keys(filter)
        self._query_vizier(center, radius, table=self.vizier_table)

        flux = np.array(self._last_query_table[flux_key].data)
        if flux_error_key is not None:
            flux_error = np.array(self._last_query_table[flux_error_key].data)
        else:
            flux_error = np.array([np.nan]*len(flux))

        return flux, flux_error

    def match_objects(self, ra, dec, filter=None, limit_angle='2 arcsec'):
        c_ra, c_dec, radius = get_center_radius(ra, dec)
        center = (c_ra, c_dec)
        c_id = self.query_id(center, radius)
        c_ra, c_dec = self.query_ra_dec(center, radius)
        if filter is not None:
            c_flux, c_flue = self.query_flux(center, radius, filter)
        else:
            c_flux = np.zeros(len(c_ra))
            c_flue = np.zeros(len(c_ra))
            c_flux.fill(np.nan)
            c_flue.fill(np.nan)

        indexes = match_indexes(ra, dec, c_ra, c_dec, limit_angle)

        m_id = np.array([c_id[i] if i != -1 else '' for i in indexes])
        m_ra = np.array([c_ra[i] if i != -1 else np.nan for i in indexes])
        m_dec = np.array([c_ra[i] if i != -1 else np.nan for i in indexes])
        m_flux = np.array([c_flux[i] if i != -1 else np.nan for i in indexes])
        m_flue = np.array([c_flue[i] if i != -1 else np.nan for i in indexes])

        return np.array(list(zip(m_id, m_ra, m_dec, m_flux, m_flue)),
                        dtype=np.dtype([('id', m_id.dtype,
                                         'ra', m_ra.dtype,
                                         'dec', m_dec.dtype,
                                         'flux', m_flux.dtype,
                                         'flux_error', m_flue.dtype)]))


def simbad_query_id(ra, dec, limit_angle, name_order=['NAME', 'HD', 'HR',
                                                      'HYP', 'TYC', 'AAVSO'],
                    tried=0):
    '''Query a single id from Simbad'''
    s = Simbad()
    q = _timeout_retry(s.query_region, center=SkyCoord(ra, dec,
                                                       unit=(u.deg, u.deg)),
                       radius=limit_angle)

    if q is not None:
        name = string_fix(q['MAIN_ID'][0])
        ids = _timeout_retry(s.query_objectids, name)['ID']
        for i in name_order:
            for k in ids:
                if i+' ' in k:
                    r = k.strip(' ').strip('NAME')
                    while '  ' in r:
                        r.replace('  ', ' ')
                    return r
    return None


class SimbadCatalog(_BasePhotometryCatalog):
    """Base class to handle with Simbad."""
    id_key = 'MAIN_ID'
    ra_key = 'RA'
    dec_key = 'DEC'
    flux_key = 'FLUX_{filter}'
    flux_error_key = 'FLUX_ERROR_{filter}'
    flux_unit_key = 'FLUX_UNIT_{filter}'
    flux_bibcode_key = 'FLUX_BIBCODE_{filter}'
    type = 'online'
    prepend_id_key = False

    def _get_simbad(self):
        s = Simbad()
        s.ROW_LIMIT = 0
        return s

    def match_object_ids(self, ra, dec, limit_angle='2 arcsec',
                         name_order=['NAME', 'HD', 'HR', 'HYP', 'TYC',
                                     'AAVSO']):
        """Get the id from Simbad for every object in a RA, Dec list."""
        # Perform it in parallel to handle the online query overhead
        p = Pool(MAX_PARALLEL_QUERY)
        results = p.map(simbad_query_id, [(r, d, limit_angle, name_order)
                                          for r, d in zip(ra, dec)])
        return results

    def _get_center(self, center):
        return get_center_skycoord(center)

    def _get_center_object(self, center):
        # we assume that every string not skycoord is a name...
        try:
            self._get_center(center)
        except ValueError:
            if isinstance(center, six.string_types):
                return center

        raise ValueError('Center {} is not a object center name for Simbad!'
                         .format(center))

    def _simbad_query_region(self, center, radius, filter=None):
        query_info = {'radius': radius, 'center': center, 'filter': filter}
        if query_info == self._last_query_info:
            logger.debug("Loading cached query.")
            return

        logger.info("Performing Simbad query with: center:{} radius:{}"
                    " filter:{}".format(center, radius, filter))

        self.flush()
        self._last_query_info = query_info

        s = self._get_simbad()
        if filter is not None:
            s.add_votable_fields('fluxdata({filter})'.format(filter=filter))

        center = self._get_center(center)
        radius = self._get_radius(radius)
        self._last_query_table = _timeout_retry(s.query_region, center, radius)

        return copy.copy(self._last_query_table)

    def query_object(self, center, filter):
        s = self._get_simbad()
        center = self._get_center_object(center)
        if filter is not None:
            s.add_votable_fields('fluxdata({filter})'.format(filter=filter))
        return _timeout_retry(s.query_object(center))

    def query_region(self, center, radius, filter=filter):
        return self._simbad_query_region(center, radius, filter)


# TODO: Continue Simbad implementation
