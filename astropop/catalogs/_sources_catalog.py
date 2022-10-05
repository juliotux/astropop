# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Base classes for astronomical catalogs queries."""

import copy
import abc
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky, Angle

from ..math.physical import QFloat
from ..logger import logger
from ._online_tools import astroquery_radius, \
                           astroquery_skycoord


def _match_indexes(ra, dec, cat_skycoords, limit_angle):
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
    # TODO: make code smarter with bayesian approach and magnitude matching

    # Matching using astropy's skycoord can be slow for big catalogs
    ind, dist, _ = match_coordinates_sky(SkyCoord(ra, dec, unit=('degree',
                                                                 'degree'),
                                                  frame='icrs'),
                                         cat_skycoords)

    index = np.zeros(len(ra), dtype=np.int)
    index.fill(-1)   # a nan index

    lim = Angle(limit_angle)
    for k in range(len(ra)):
        if dist[k] <= lim:
            index[k] = ind[k]

    logger.debug('Matched %s objects in catalog from %s total',
                 np.sum(index != -1), len(ra))

    return index


class SourcesCatalog:
    """Manage and query a catalog of point sources objects.

    This catalog wraps around `~astropy.coordinates.SkyCoord` and it's
    query mechanism, extending it to include sources id or names,
    magnitudes or other informtions that user may want. It's initialization
    accept all initialization arguments from this class. Other informations
    are stored according the following arguments.

    Currently, it is designed to contain only magnitudes for photometric
    data.

    Parameters
    ----------
    ids: array
        Names or ids of the objects in the catalog.
    mag: `dict` (optional)
        Dictionary of photometric magnitudes for each available filter. The
        keys are the names of the filters and the values are the photometric
        magnitudes of the object in a `astropop.math.QFloat` array. Photometry
        will be only available if this argument is set.
    *args, **kwargs:
        Arguments to be passed to `~astropy.coordinates.SkyCoord`
        initialization. See `~astropy.coordinates.SkyCoord` docs for more
        details.
        ra, dec: array (optional)
            RA and DEC coordinates of the object. Conflicts with ``coords``
            argument. If floats, are interpreted as decimal degrees.
        pm_ra_cosdec, pm_dec: `~astropy.units.Quantity` (optional)
            Proper motion of both coordinates.
        unit: `~astropy.units.Unit`, string, or tuple
            Units for supplied coordinate values.
        obstime: time-like (optional)
            Time of observation of the values. Used to compute proper
            motion on a target observation time.
        frame: str (optional)
            Celestial frame of coordinates. Default is 'ICRS'
    """

    _base_table = None  # Store array of ids
    _mags_table = None  # Store magnitudes with filter colname in QFloat format

    def __init__(self, *args, ids=None, mag=None,
                 query_table=None, **kwargs):
        self._base_table = Table()
        # initializate coords and skycoords using default kwargs.
        coords = SkyCoord(*args, **kwargs, copy=True)

        # IDs are stored in a numpy 1d-array
        if len(np.shape(ids)) != 1:
            raise ValueError('Sources ID must be a 1d array.')

        if len(ids) != len(coords):
            raise ValueError('Sources IDs and coordinates must have the same '
                             'number of elements.')

        # initialize store table
        self._base_table['id'] = np.array(ids)
        self._base_table['coords'] = coords

        # magnitudes are stored in a dict of QFloat
        if not isinstance(mag, dict):
            raise TypeError('mag must be a dictionary of magnitudes')
        for i in mag.keys():
            if len(self._coords[i]) != len(mag[i]):
                raise ValueError('Lengths of magnitudes must be the same as '
                                 'the number of sources.')
            self._add_mags(i, mag[i])

        # Store base query table
        if query_table is not None:
            self._query = query_table

    def _add_mags(self, band, mag):
        """Add a mag to the magnitude dict."""
        # initialize magnitude if not initializated.
        if self._mags is None:
            self._mags = Table()
        # add magnitude to the dict
        m = QFloat(mag)
        if len(m) != len(self._base_table):
            raise ValueError('Lengths of magnitudes must be the same as '
                             'the number of sources.')
        self._mags[f'{band}'] = m.nominal
        self._mags[f'{band}_error'] = m.std_dev

    def sources_id(self):
        """Get the list of sources id in catalog."""
        return self._base_table['id'].value

    def skycoord(self):
        """Get the sources coordinates in SkyCoord format."""
        return self.get_coordinates()

    def ra_dec_list(self):
        """Get the sources coordinates in [(ra, dec)] format."""
        sk = self.skycoord
        try:
            return np.array(list(zip(sk.ra.degree, sk.dec.degree)))
        except TypeError:
            return [(sk.ra.degree, sk.dec.degree)]

    def magnitude(self, band):
        """Get the sources magnitude in QFloat format.

        Parameters
        ----------
        band : str
            The band name.

        Returns
        -------
        mag : float
            The sources magnitude in QFloat format.
        """
        return QFloat(self._mags[f'{band}'], self._mags[f'{band}_error'],
                      'mag')

    def mag_list(self, band):
        """Get the sources photometric mag in [(mag, mag_error)] format.

        Parameters
        ----------
        band : str
            The band name.

        Returns
        -------
        mag_list : list
            List of tuples of (mag, mag_error).
        """
        if self._mags is None:
            return
        return list(zip(self._mags[f'{band}'], self._mags[f'{band}_error']))

    def copy(self):
        """Copy the current catalog to a new instance."""
        return copy.copy(self)

    def get_coordinates(self, obstime=None):
        """Get the skycoord positions from the catalog."""
        sk = self._base_table['coords']
        try:
            return sk.apply_space_motion(new_obstime=obstime)
        except ValueError:
            return copy.copy(sk)

    def match_objects(self, ra, dec, limit_angle, obstime=None, table=False):
        """Match a list of ra, dec objects to this catalog.

        Parameters
        ----------
        ra, dec: float or array-like
            RA and Dec coordinates of the objects to be matched to this
            catalog. All coordinates in decimal degree format.
        limit_angle: string, float, `~astropy.coordinates.Angle`
            Angle limit for matching indexes. If string, if must be
            `~astropy.coordinates.Angle` compatible. If float, it will be
            interpreted as a decimal degree.
        obstime: `~astropy.time.Time` (optional)
            Observation time. If passed, it will be used to apply the proper
            motion to the coordinates, if available.
            Default: None
        table: bool (optional)
            Return a table instead a source catalog.
            Default: False
        """
        cat_sk = self.get_coordinates(obstime=obstime)
        indexes = _match_indexes(ra, dec, cat_sk,
                                 astroquery_radius(limit_angle))

        raise NotImplementedError

        if table:
            return ncat.table
        return ncat

    def __getitem__(self, item):
        """Get items from the catalog.

        A new catalog with only the selected sources is returned.
        If item is a string, a column from the result query will be returned.
        """
        if isinstance(item, str):
            if self._query is None:
                raise KeyError('Empty query.')
            return copy.copy(self._query[item])

        if not isinstance(item, (int, list, np.ndarray, slice)):
            raise KeyError(f"{item}")

        if isinstance(item, int):
            item = [item]

        nc = SourcesCatalog.__new__()
        nc._base_table = self._base_table[item]
        if self._mags is not None:
            nc._mags = self._mags[item]
        return nc

    def __len__(self):
        return len(self._ids)


class _OnlineSourcesCatalog(SourcesCatalog, abc.ABC):
    """Sources Catalog based on online queries."""

    _query = None

    def __init__(self, center, radius):
        """Query the catalog and create the source catalog instance.

        Parameters
        ----------
        center: string, tuple or `~astropy.coordinates.SkyCoord`
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
        """
        self._center = astroquery_skycoord(center)
        self._radius = astroquery_radius(radius)

        # setup the catalog if needed
        self._setup_catalog()

        # perform the query
        logger.info('Quering region centered at %s with radius %s',
                    self._center, self._radius)
        self._do_query()

    @abc.abstractmethod
    def _setup_catalog(self):
        """If a catalog setup is needed."""

    @abc.abstractmethod
    def _do_query(self):
        """Query the catalog. Must end with the catalog initialization."""

    @property
    def query_colnames(self):
        """Get column names from query"""
        return copy.copy(self._query.colnames)

    @property
    def available_filters(self):
        """List available filters for the catalog."""
        return copy.copy(self._available_filters)
