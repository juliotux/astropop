# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Module to handle online catalog queries."""

import six
import copy
import numpy as np
from multiprocessing.pool import Pool
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u

from .base_catalog import _BasePhotometryCatalog, match_indexes
from ..logger import logger
from ..astrometry.coords_utils import guess_coordinates
from ..py_utils import string_fix, process_list


MAX_PARALLEL_QUERY = 30
MAX_RETRIES_TIMEOUT = 10

__all__ = ['VizierCatalogClass', 'SimbadCatalogClass', 'UCAC5Catalog',
           'SimbadCatalog', 'UCAC4Catalog', 'GSC23Catalog', 'APASSCalatolg',
           'DENISCatalog', 'TWOMASSCatalog', 'default_catalogs']


def _timeout_retry(func, *args, **kwargs):
    tried = kwargs.pop('_____retires', 0)
    try:
        q = func(*args, **kwargs)
    except TimeoutError:
        if tried >= MAX_RETRIES_TIMEOUT:
            logger.warning('TimeOut obtained in %s tries, aborting.',
                        MAX_RETRIES_TIMEOUT)
            return
        return _timeout_retry(func, *args, **kwargs, _____retires=tried+1)
    return q


def _wrap_query_table(func):
    """Wrap queries to return bytes columns as strings."""
    def wrapper(*args, **kwargs):
        table = func(*args, **kwargs)
        for i in table.columns:
            if table[i].dtype == np.dtype('O'):
                table[i] = process_list(string_fix, table[i])
        return table
    return wrapper


def get_center_radius(ra, dec):
    """Get the center of a list RA and DEC."""
    center_ra = (np.max(ra) + np.min(ra))/2
    center_dec = (np.max(dec) + np.min(dec))/2
    radius = np.max([np.max(ra) - np.min(ra),
                     np.max(dec) - np.min(dec)])
    return center_ra, center_dec, radius


def get_center_skycoord(center):
    if isinstance(center, six.string_types):
        try:
            return SkyCoord(center)
        except (ValueError):
            t = Simbad.query_object(center)
            if len(t) == 0:
                raise ValueError(f'Coordinates {center} could not be'
                                 ' resolved.')
            return guess_coordinates(t['RA'][0], t['DEC'][0], skycoord=True)
    elif isinstance(center, (tuple, list, np.ndarray)) and len(center) == 2:
        return guess_coordinates(center[0], center[1], skycoord=True)
    elif isinstance(center, SkyCoord):
        return center

    raise ValueError(f'Center coordinates {center} not undertood.')


class VizierCatalogClass(_BasePhotometryCatalog):
    """Base class to handle with Vizier online catalogs."""

    vizier_table = None
    id_key = None
    ra_key = 'RAJ2000'
    dec_key = 'DEJ2000'
    flux_key = None
    flux_error_key = None
    flux_unit = None
    available_filters = None
    type = 'online'
    prepend_id_key = False
    bibcode = None
    _valid_init_kwargs = set(['vizier_table', 'id_key', 'ra_key', 'dec_key',
                              'flux_key', 'flux_error_key', 'flux_unit',
                              'prepend_id_key', 'available_filters',
                              'bibcode', 'comment'])
    _last_query_info = None
    _last_query_table = None

    def __init__(self, **kwargs):
        self.vizier = Vizier()
        self.vizier.ROW_LIMIT = -1
        # self.vizier.VIZIER_SERVER = 'vizier.cfa.harvard.edu'

        for i, v in kwargs.items():
            if i in self._valid_init_kwargs:
                self.__setattr__(i, v)
            else:
                raise ValueError('Invalid parameter {} passed to'
                                 ' VizierCatalogClass')

    def _flux_keys(self, band):
        flux_key = self.flux_key.format(band=band)
        if self.flux_error_key is not None:
            flux_error_key = self.flux_error_key.format(band=band)
        else:
            flux_error_key = None
        return flux_key, flux_error_key

    def _get_center(self, center):
        return get_center_skycoord(center, logger)

    def _query_vizier(self, center, radius, table):
        """Query in vizier site."""
        # check if the new query is equal to previous one. Only perform a new
        # query if it is not redundant
        query_info = {'radius': radius, 'center': center, 'table': table}
        if query_info == self._last_query_info:
            logger.debug("Loading cached query.")
            return copy.copy(self._last_query_table)

        if table is None:
            raise ValueError("No Vizier table was defined.")

        logger.info("Performing Vizier query with: center:%s "
                    "radius:%s vizier_table:%s",
                    center, radius, table)

        self._last_query_info = query_info
        self._last_query_table = None

        center = self._get_center(center)

        if radius is not None:
            radius = f"{self._get_radius(radius)}d"
            query = _timeout_retry(self.vizier.query_region, center,
                                   radius=radius, catalog=table)
        else:
            query = _timeout_retry(self.vizier.query_object, center,
                                   catalog=table)
        if len(query) == 0:
            raise RuntimeError("No online catalog results were found.")
        self._last_query_table = query[0]
        return copy.copy(self._last_query_table)

    def query_object(self, center, **kwargs):
        """Query a single object in the catalog."""
        return self._query_vizier(center, radius=None, table=self.vizier_table)

    def query_region(self, center, radius, **kwargs):
        """Query all objects in a region."""
        return self._query_vizier(center, radius, table=self.vizier_table)

    def query_ra_dec(self, center, radius, **kwargs):
        """Query coordinates in a region of the catalog."""
        if self.ra_key is None or self.dec_key is None:
            raise ValueError("Invalid RA or Dec keys.")
        self._query_vizier(center, radius, table=self.vizier_table)
        ra = self._last_query_table[self.ra_key].data
        dec = self._last_query_table[self.dec_key].data

        # Solve the most common types of coordinates
        coords = guess_coordinates(ra, dec)
        ra = np.array(coords.ra.degree)
        dec = np.array(coords.dec.degree)

        return ra, dec

    def query_id(self, center, radius, **kwargs):
        """Query the object name in a region of the catalog."""
        if self.id_key is None:
            raise ValueError("Invalid ID key.")
        self._query_vizier(center, radius, table=self.vizier_table)

        if self.id_key == -1:
            return np.array(['']*len(self._last_query_table))

        idn = self._last_query_table[self.id_key].data
        if self.prepend_id_key:
            if isinstance(self.prepend_id_key, six.string_types):
                id_key = self.prepend_id_key
            else:
                id_key = self.id_key
            idn = [f"{id_key} {i}" for i in idn]
            idn = np.array(idn)

        return string_fix(idn)

    def query_flux(self, center, radius, band, **kwargs):
        """Query the flux data in a region of the catalog."""
        self.check_filter(band)
        flux_key, flux_error_key = self._flux_keys(band)
        self._query_vizier(center, radius, table=self.vizier_table,)

        flux = np.array(self._last_query_table[flux_key].data)
        try:
            flux_error = np.array(self._last_query_table[flux_error_key].data)
        except KeyError:
            flux_error = np.array([np.nan]*len(flux))

        return flux, flux_error

    def match_objects(self, ra, dec, filter=None, limit_angle='2 arcsec'):
        """Match objects from RA DEC list with this catalog."""
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
                        dtype=np.dtype([('id', m_id.dtype),
                                        ('ra', m_ra.dtype),
                                        ('dec', m_dec.dtype),
                                        ('flux', m_flux.dtype),
                                        ('flux_error', m_flue.dtype)]))


def simbad_query_id(ra, dec, limit_angle, name_order=None):
    """Query a single id from Simbad."""
    if name_order is None:
        name_order = ['NAME', 'HD', 'HR', 'HYP', 'TYC', 'AAVSO']

    s = Simbad()
    q = _timeout_retry(s.query_region, center=SkyCoord(ra, dec,
                                                       unit=(u.degree,
                                                             u.degree)),
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

    def _get_simbad(self):
        s = Simbad()
        s.ROW_LIMIT = 0
        return s

    def _get_center(self, center):
        c = get_center_skycoord(center)
        return c

    def _get_center_object(self, center):
        # we assume that every string not skycoord is a name...
        try:
            self._get_center(center)
        except ValueError:
            if isinstance(center, six.string_types):
                return center

        raise ValueError(f'Center {center} is not a object center name'
                         ' for Simbad!')

    def _flux_keys(self, band):
        flux_key = self.flux_key.format(band=band)
        flux_error_key = self.flux_error_key.format(band=band)
        flux_bibcode_key = self.flux_bibcode_key.format(band=band)
        flux_unit_key = self.flux_unit_key.format(band=band)
        return flux_key, flux_error_key, flux_unit_key, flux_bibcode_key

    def query_object(self, center, band=None, **kwargs):
        """Query a single object in the catalog."""
        s = self._get_simbad()
        # query object should not need this
        # center = self._get_center_object(center)
        if band is not None:
            s.add_votable_fields(f'fluxdata({band})')
        return _timeout_retry(_wrap_query_table(s.query_object),
                              center, **kwargs)

    def query_region(self, center, radius, band=None, **kwargs):
        """Query all objects in a region."""
        query_info = {'radius': radius, 'center': center, 'band': band}
        if query_info == self._last_query_info:
            logger.debug("Loading cached query.")
            return copy.copy(self._last_query_table)

        logger.info("Performing Simbad query with: center: %s"
                    "radius:%s band:%s",
                    center, radius, band)

        self._last_query_info = query_info
        self._last_query_table = None

        s = self._get_simbad()
        if band is not None:
            s.add_votable_fields(f'fluxdata({band})')

        center = self._get_center(center)
        radius = f"{self._get_radius(radius)}d"
        query = _timeout_retry(_wrap_query_table(s.query_region),
                               center, radius)
        self._last_query_table = query
        return copy.copy(self._last_query_table)

    def query_ra_dec(self, center, radius, **kwargs):
        """Query coordinates in a region of the catalog."""
        if self.ra_key is None or self.dec_key is None:
            raise ValueError("Invalid RA or Dec keys.")
        self.query_region(center, radius)
        ra = self._last_query_table[self.ra_key].data
        dec = self._last_query_table[self.dec_key].data

        # Solve the most common types of coordinates
        coords = guess_coordinates(ra, dec)
        ra = np.array(coords.ra.degree)
        dec = np.array(coords.dec.degree)

        return ra, dec

    def query_id(self, center, radius, **kwargs):
        """Query object names in a region of the catalog."""
        if self.id_key is None:
            raise ValueError("Invalid ID key.")
        self.query_region(center, radius, band=None)

        if self.id_key == -1:
            return np.array(['']*len(self._last_query_table))

        idn = self._last_query_table[self.id_key].data
        if self.prepend_id_key:
            idn = [f"{self.id_key} {i}" for i in idn]
            idn = np.array(idn)

        return string_fix(idn)

    def query_flux(self, center, radius, band,
                   return_bibcode=False, **kwargs):
        """Query the flux data in a region of the catalog."""
        self.check_filter(band)
        flux_key, error_key, unit_key, bibcode_key = self._flux_keys(band)
        self.query_region(center, radius, band=band)

        flux = np.array(self._last_query_table[flux_key].data)
        if error_key is not None:
            flux_error = np.array(self._last_query_table[error_key].data)
        else:
            flux_error = np.array([np.nan]*len(flux))
        if bibcode_key is not None:
            bibcode = np.array(self._last_query_table[bibcode_key].data)
        else:
            bibcode = np.zeros(len(flux), dtype=str)
        if unit_key is not None:
            unit = np.array(self._last_query_table[unit_key].data)
        else:
            unit = np.zeros(len(flux), dtype=str)

        if return_bibcode:
            return flux, flux_error, unit, bibcode
        return flux, flux_error, unit

    def match_objects(self, ra, dec, band=None, limit_angle='2 arcsec'):
        """Match objects from RA DEC list with this catalog."""
        c_ra, c_dec, radius = get_center_radius(ra, dec)
        center = (c_ra, c_dec)
        c_id = self.query_id(center, radius)
        c_ra, c_dec = self.query_ra_dec(center, radius)
        if band is not None:
            c_flux, c_flue, c_unit, c_flub = self.query_flux(center, radius,
                                                             band, True)
        else:
            c_flux = np.zeros(len(c_ra))
            c_flue = np.zeros(len(c_ra))
            c_flub = np.zeros(len(c_ra), dtype=str)
            c_unit = np.zeros(len(c_ra), dtype=str)
            c_flux.fill(np.nan)
            c_flue.fill(np.nan)

        indexes = match_indexes(ra, dec, c_ra, c_dec, limit_angle)

        m_id = np.array([c_id[i] if i != -1 else '' for i in indexes])
        m_ra = np.array([c_ra[i] if i != -1 else np.nan for i in indexes])
        m_dec = np.array([c_ra[i] if i != -1 else np.nan for i in indexes])
        m_flux = np.array([c_flux[i] if i != -1 else np.nan for i in indexes])
        m_flue = np.array([c_flue[i] if i != -1 else np.nan for i in indexes])
        m_flub = np.array([c_flub[i] if i != -1 else '' for i in indexes])
        m_unit = np.array([c_unit[i] if i != -1 else '' for i in indexes])

        return np.array(list(zip(m_id, m_ra, m_dec, m_flux, m_flue, m_unit,
                                 m_flub)),
                        dtype=np.dtype([('id', m_id.dtype),
                                        ('ra', m_ra.dtype),
                                        ('dec', m_dec.dtype),
                                        ('flux', m_flux.dtype),
                                        ('flux_error', m_flue.dtype),
                                        ('flux_unit', m_unit.dtype),
                                        ('flux_bibcode', m_flub.dtype)]))

    def match_object_ids(self, ra, dec, limit_angle='2 arcsec',
                         name_order=['NAME', 'HD', 'HR', 'HYP', 'TYC',
                                     'AAVSO']):
        """Get the id from Simbad for every object in a RA, Dec list."""
        # Perform it in parallel to handle the online query overhead
        p = Pool(MAX_PARALLEL_QUERY)
        results = p.map(simbad_query_id, [(r, d, limit_angle, name_order)
                                          for r, d in zip(ra, dec)])
        return results


###############################################################################
# catalogs
###############################################################################

SimbadCatalog = SimbadCatalogClass()

# TODO: For UCAC4 and APASS, try convert g, r, i to R band!
UCAC4Catalog = VizierCatalogClass(available_filters=["B", "V", "g", "r", "i"],
                                  vizier_table="I/322A",
                                  prepend_id_key=True,
                                  id_key="UCAC4",
                                  ra_key="RAJ2000",
                                  dec_key="DEJ2000",
                                  flux_key="{band}mag",
                                  flux_error_key="e_{band}mag",
                                  flux_unit="mag",
                                  bibcode="2013AJ....145...44Z",
                                  comment="Magnitudes from APASS")


UCAC5Catalog = VizierCatalogClass(available_filters=["Gaia", "R", "f."],
                                  vizier_table="I/340",
                                  prepend_id_key=True,
                                  id_key='SrcIDgaia',
                                  ra_key="RAJ2000",
                                  dec_key="DEJ2000",
                                  flux_key="{band}mag",
                                  flux_error_key="e_{band}mag",
                                  flux_unit="mag",
                                  bibcode="2017yCat.1340....0Z",
                                  comment="Rmag from NOMAD")

APASSCalatolg = VizierCatalogClass(available_filters=["B", "V", "g'",
                                                      "r'", "i'"],
                                   vizier_table="II/336/apass9",
                                   prepend_id_key=False,
                                   id_key=-1,
                                   ra_key="RAJ2000",
                                   dec_key="DEJ2000",
                                   flux_key="{band}mag",
                                   flux_error_key="e_{band}mag",
                                   flux_unit="mag",
                                   bibcode="2016yCat.2336....0H",
                                   comment="g', r' and i' magnitudes in"
                                           " AB system")

DENISCatalog = VizierCatalogClass(available_filters=["I", "J", "K"],
                                  vizier_table="B/denis",
                                  prepend_id_key=True,
                                  id_key="DENIS",
                                  ra_key="RAJ2000",
                                  dec_key="DEJ2000",
                                  flux_key="{band}mag",
                                  flux_error_key="e_{band}mag",
                                  flux_unit="mag",
                                  bibcode="2005yCat.2263....0T",
                                  comment="Id, Jd, Kd may differ a bit "
                                          "from 2MASS catalog")

TWOMASSCatalog = VizierCatalogClass(available_filters=["I", "J", "K"],
                                    vizier_table="B/denis",
                                    prepend_id_key='2MASS',
                                    id_key="_2MASS",
                                    ra_key="RAJ2000",
                                    dec_key="DEJ2000",
                                    flux_key="{band}mag",
                                    flux_error_key="e_{band}mag",
                                    flux_unit="mag",
                                    bibcode="2003yCat.2246....0C",
                                    comment="")


class _GCS23Catalog(VizierCatalogClass):
    available_filters = ["U", "B", "Bj", "V", "R", "F", "N"]
    vizier_table = "I/305/out"
    prepend_id_key = False
    id_key = 'GSC2.3'
    ra_key = "RAJ2000"
    dec_key = "DEJ2000"
    flux_key = "{band}mag"
    flux_error_key = "e_{band}mag"
    flux_unit = "mag"
    bibcode = "2008AJ....136..735L"
    comment = "Hubble Guide Star Catalog 2.3.2 (STScI, 2006)." \
              " R magnitude was assumed to be equal to F magnitude."
    filt_conversion = {'R': 'F',
                       'Bj': 'j'}

    def _flux_keys(self, filter):
        if filter in self.filt_conversion.keys():
            filter = self.filt_conversion[filter]
        return VizierCatalogClass._flux_keys(self, filter)


GSC23Catalog = _GCS23Catalog()

default_catalogs = {'Simbad': SimbadCatalog,
                    'UCAC4': UCAC4Catalog,
                    'UCAC5': UCAC5Catalog,
                    'APASS': APASSCalatolg,
                    'DENIS': DENISCatalog,
                    '2MASS': TWOMASSCatalog,
                    'GSC2.3': GSC23Catalog}
