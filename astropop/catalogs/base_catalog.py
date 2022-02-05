# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Base classes for astronomical catalogs query."""

import copy
import abc
import numpy as np
from astropy.table import Table
from astropy.coordinates import Angle, SkyCoord, match_coordinates_sky

from ..logger import logger
from ._online_tools import _timeout_retry, get_center_radius, \
                           _wrap_query_table
from ..astrometry.coords_utils import guess_coordinates


__all__ = ['match_indexes']


def match_indexes(ra, dec, cat_ra, cat_dec, limit_angle):
    """Match list of coordinates with a calatog.

    Parameters
    ----------
    ra, dec: list of float
        List of objects coordinates to be matched in the catalog. All
        coordinates must be in decimal degrees.
    cat_ra, cat_dec: list of float
        List of catalog coordinates. All coordinates must be in decimal
        degrees.
    limit_angle: string, float, `~astropy.coordinates.Angle`
        Angle limit for matching indexes. If string, if must be
        `~astropy.coordinates.Angle` compatible. If float, it will be
        interpreted as a decimal degree.
    logger: `~logging.Logger`

    Returns
    -------
    index: `~numpy.ndarray`
        List containing the indexes in the catalog that matched the object
        coordinates. If -1, it represents objects not matched.
    """
    # # Matching using astropy's skycoord can be slow for big catalogs
    ind, dist, _ = match_coordinates_sky(SkyCoord(ra, dec, unit=('degree',
                                                                 'degree'),
                                                  frame='icrs'),
                                         SkyCoord(cat_ra, cat_dec,
                                                  unit=('degree', 'degree'),
                                                  frame='icrs'))

    index = np.zeros(len(ra), dtype=np.int)
    index.fill(-1)   # a nan index

    lim = Angle(limit_angle)
    for k in range(len(ra)):
        if dist[k] <= lim:
            index[k] = ind[k]

    logger.debug('Matched %s objects in catalog from %s total',
                 np.sum(index != -1), len(ra))

    return index


class _BaseCatalog(abc.ABC):
    """Base class for catalog query."""

    # stores all the needed information to avoid continuous redundat queries
    _last_query_info = None
    _last_query_table = None
    comment = None

    def __evaluate__(self, center, radius=None, **kwargs):
        """Query the results in the catalog.

        Parameters:
        -----------
        center: string or tuple
            The center of the search field.
            If center is a string, can be an object name or the string
            containing the object coordinates. If it is a tuple, have to be
            (ra, dec) coordinates, in hexa or decimal degrees format.
        radius: string, float, `~astropy.coordinates.Angle`
                or None (optional)
            The radius to search. If None, the query will be performed as
            single object query mode. Else, the query will be performed as
            field mode. If a string value is passed, it must be readable by
            astropy.coordinates.Angle. If a float value is passed, it will
            be interpreted as a decimal degree radius.
        ** kwargs
            other arguments to be passed to the catalog functions.
        """
        if radius is None:
            return self.query_object(center, **kwargs)
        return self.query_region(center, radius, **kwargs)

    def flush(self):
        """Clear previous query informations."""
        del self._last_query_info
        del self._last_query_table

    def copy(self):
        """Get a new copy of the catalog."""
        return copy.deepcopy(self)


class _BasePhotometryCatalog(_BaseCatalog, abc.ABC):
    """A base class for photometry catalogs."""

    name = None
    type = None
    flux_unit = None
    available_filters = []
    bibcode = None

    def check_filter(self, band, raise_error=True):
        """Check if a filter is available for this catalog."""
        if not raise_error:
            return band in self.available_filters
        if band in self.available_filters:
            return True

        raise ValueError(f'This catalog does not support {band} filter. '
                         'The available formats are:'
                         f' {self.available_filters}')

    def filter_ra_dec(self, query):
        """Filter coordinates in a query result."""
        if self.ra_key is None or self.dec_key is None:
            raise ValueError("Invalid RA or Dec keys.")

        if query is None:
            query = self._last_query_table

        ra = query[self.ra_key].data
        dec = query[self.dec_key].data

        # Solve the most common types of coordinates
        coords = guess_coordinates(ra, dec)
        ra = np.array(coords.ra.degree)
        dec = np.array(coords.dec.degree)

        return ra, dec

    def filter_id(self, query=None):
        """Filter object names in a query result."""
        if self.id_key is None:
            raise ValueError("Invalid ID key.")

        if query is None:
            query = self._last_query_table

        if self.id_key == -1:
            return np.array(['']*len(query))

        idn = query[self.id_key].data
        return self._id_resolve(idn)

    def _query(self, querier, *args, **kwargs):
        """Perform a query. All args are passed to querier."""
        query_info = (args, kwargs)
        if (query_info == self._last_query_info and
           self._last_query_table is not None):
            logger.debug("Loading cached query.")
            return copy.copy(self._last_query_table)

        logger.info("Performing %s query with parameters: %s",
                    self.__class__.__name__, query_info)

        self._last_query_info = query_info
        self._last_query_table = None

        query = _timeout_retry(querier, *args, **kwargs)
        if query is None:
            raise RuntimeError("No online catalog result found.")
        self._last_query_table = _wrap_query_table(query)
        return copy.copy(self._last_query_table)

    def _match_objects(self, ra, dec, band, limit_angle,
                       flux_keys, table_props):
        c_ra, c_dec, radius = get_center_radius(ra, dec)
        query = self.query_region((c_ra, c_dec), radius)
        cat = Table()
        cat['id'] = self.filter_id(query)
        cat['ra'], cat['dec'] = self.filter_ra_dec(query)

        indexes = match_indexes(ra, dec, cat['ra'], cat['dec'], limit_angle)
        if band is not None:
            for k, v in zip(flux_keys, self.filter_flux(band, query)):
                cat[k] = v
        else:
            for k in flux_keys:
                cat[k] = np.full(len(cat), fill_value=np.nan)

        res = Table()
        res['ra'] = ra
        res['dec'] = dec
        for k, empty in table_props:
            res['cat_'+k] = [cat[k][i] if i < 0 else empty for i in indexes]

        return res
