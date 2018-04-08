# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

from ..astrometry.coords_utils import guess_coordinates
from .base_catalog import _BasePhotometryCatalog


class _LocalCatalog(_BasePhotometryCatalog):
    type = 'local'
    id_key = None
    ra_key = None
    dec_key = None
    flux_key = None
    flux_error_key = None
    flux_unit = None
    available_filters = None
    prepend_id_key = False
    bibcode = None

    _table = None  # Where the data is stored.

    @property
    def skycoords(self):
        if self.ra_key is not None and self.dec_key is not None:
            tabs = [guess_coordinates(self._table[self.ra_key][i],
                                      self._table[self.dec_key][i])
                    for i in range(len(self._table))]
            return SkyCoord([s.ra.degree for s in tabs],
                            [s.dec.degree for s in tabs],
                            unit=('degree', 'degree'))
        else:
            return None

    @property
    def id(self):
        if self.id_key is None:
            return None
        return np.array(self._table[self.id_key])

    def _query_index(self, center=None, radius=None):
        """Return the index of the points within a radius distance of center
        point.

        If center is None, return all
        """
        if center is None:
            return np.arange(0, len(self._talbe), 1)

        center = self._get_center(center)
        radius = self._get_radius(radius)
        coords = self.skycoords
        sep = coords.separation(center)
        filt = np.where(sep <= radius*u.degree)
        return filt

    def query_ra_dec(self, center=None, radius=None):
        """Query coordinates in a region of the catalog."""
        filt = self._query_index(center, radius)
        coords = self.skycoords
        return coords.ra.degree[filt], coords.dec.degree[filt]

    def query_flux(self, center=None, radius=None):
        """Query the flux data in a region of the catalog."""
        filt = self._query_index(center, radius)
        if self.flux_key is not None:
            flux = np.array(self._table[self.flux_key][filt])
        else:
            flux = np.zeros(len(filt))
            flux = flux.fill(np.nan)
        if self.flux_error_key is not None:
            error = np.array(self._table[self.flux_error_key][filt])
        else:
            error = np.zeros(len(filt))
            error = flux.fill(np.nan)
        return flux, error

    def query_id(self, center=None, radius=None):
        """Query coordinates in a region of the catalog."""
        filt = self._query_index(center, radius)
        return self.id[filt]

    def match_objects(self, ra, dec, limit_angle='2 arcsec'):
        '''Query the informations in the catalog from a list of ra and dec
        coordinates, matching the stars by a limit_angle.
        '''
        rac, decc = self.query_ra_dec()
        coords = SkyCoord(rac, decc, unit=('degree', 'degree'))

        indx, sep2, sep3 = coords.match_coordinates_sky(ra, dec)
        filt = sep2 <= self._get_radius(limit_angle)*u.degree

        nstars = len(ra)
        m_id = self.query_id()
        m_id = [m_id[i] if filt[i] else '' for i in range(nstars)]
        m_ra, m_dec = self.query_ra_dec()
        m_ra = [m_ra[i] if filt[i] else np.nan for i in range(nstars)]
        m_dec = [m_dec[i] if filt[i] else np.nan for i in range(nstars)]
        try:
            flux, error = self.query_flux()
            m_f = [flux[i] if filt[i] else np.nan for i in range(nstars)]
            m_e = [error[i] if filt[i] else np.nan for i in range(nstars)]
        except NotImplementedError:
            m_f = np.zeros(nstars)
            m_f.fill(np.nan)
            m_e = np.zeros(nstars)
            m_e.fill(np.nan)

        return np.array(list(zip(m_id, m_ra, m_dec, m_f, m_e)),
                        dtype=np.dtype([('id', m_id.dtype),
                                        ('ra', m_ra.dtype),
                                        ('dec', m_dec.dtype),
                                        ('flux', m_f.dtype),
                                        ('flux_error', m_e.dtype)]))

    def match_object_ids(self, ra, dec, limit_angle='2 arcsec'):
        return self.match_objects(ra, dec, limit_angle=limit_angle)['id']


class ASCIICatalogClass(_LocalCatalog):
    type = 'local'

    def __init__(self, filename, id_key=None, ra_key=None, dec_key=None,
                 flux_key=None, flux_error_key=None, flux_unit=None,
                 available_filters=None, prepend_id_key=False, bibcode=None,
                 **reader_kwargs):
        """
        **reader_kwargs : kwargs to be passed to the Table.read function
        """
        self._table = Table.read(filename, **reader_kwargs)

        self.id_key = id_key
        self.ra_key = ra_key
        self.dec_key = dec_key
        self.flux_key = flux_key
        self.flux_error_key = flux_error_key
        self.flux_unit = flux_unit
        self.available_filters = available_filters
        self.prepend_id_key = prepend_id_key
        self.bibcode = bibcode
