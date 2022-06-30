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
    # # Matching using astropy's skycoord can be slow for big catalogs
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
    ids: array (optional)
        Names or ids of the objects in the catalog.
    mag: array or `~astropop.math.QFloat` (optional)
        Photometric magnitude of the object. If a list
        or array, must be 1-dimensional array containing only the magnitudes.
        If QFloat, ``mag_error`` and ``mag_unit`` arguments
        will be ignored. Photometry will be only available if this
        argument is set.
    mag_error: array (optional)
        Photometric magnitude errors in the same unit of the `mag`
        argument. Ignored if ``mag`` is a QFloat.
    mag_unit: str, `~astropy.units.Unit` or array (optional)
        Unit of the photometric magnitude.
        Ignored if ``mag`` is a QFloat.
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

    _ids = None  # Store array of ids
    _mags = None  # Store QFloat of mags
    _coords = None  # Store Skycoord of coordinates

    def __init__(self, *args, ids=None, mag=None, mag_error=None,
                 mag_unit=None, **kwargs):
        # initializate coords and skycoords using default kwargs.
        self._coords = SkyCoord(*args, **kwargs, copy=True)

        # IDs are stored in a numpy 1d-array
        if len(np.shape(ids)) != 1:
            raise ValueError('Sources ID must be a 1d array.')
        self._ids = np.array(ids)

        # magnitudes are stored as QFloat
        if isinstance(mag, QFloat):
            self._mags = mag
        elif mag is not None:
            self._mags = QFloat(mag, uncertainty=mag_error, unit=mag_unit)
        else:
            self._mags = None

    @property
    def sources_id(self):
        """Get the list of sources id in catalog."""
        return copy.copy(self._ids)

    @property
    def skycoord(self):
        """Get the sources coordinates in SkyCoord format."""
        return self.get_coordinates()

    @property
    def ra_dec_list(self):
        """Get the sources coordinates in [(ra, dec)] format."""
        sk = self.skycoord
        try:
            return np.array(list(zip(sk.ra.degree, sk.dec.degree)))
        except TypeError:
            return (sk.ra.degree, sk.dec.degree)

    @property
    def magnitude(self):
        """Get the sources magnitude in QFloat format."""
        return copy.copy(self._mags)

    @property
    def mag_list(self):
        """Get the sources photometric mag in [(mag, mag_error)] format."""
        if self._mags is None:
            return
        try:
            return np.array(list(zip(self._mags.nominal,
                                     self._mags.uncertainty)))
        except TypeError:
            return (self._mags.nominal, self._mags.uncertainty)

    @property
    def table(self):
        """Get the soures id, coordinates and flux in Table format."""
        sk = self.skycoord
        if self.magnitude is None:
            t = Table({'id': self.sources_id,
                       'ra': sk.ra.degree,
                       'dec': sk.dec.degree},
                      units=(None, 'degree', 'degree'))
        else:
            fl = self.magnitude
            t = Table({'id': self.sources_id,
                       'ra': sk.ra.degree,
                       'dec': sk.dec.degree,
                       'mag': fl.nominal,
                       'mag_error': fl.uncertainty},
                      units=(None, 'degree', 'degree', 'mag', 'mag'))

        return t

    @property
    def array(self):
        """Get the soures id, coordinates and flux in ndarray format."""
        return self.table.as_array()

    @property
    def center(self):
        """Get the center of the catalog query, in SkyCoord format."""
        return copy.copy(self._center)

    @property
    def radius(self):
        """Get the radius of the catalog query, in Angle format."""
        return copy.copy(self._radius)

    @property
    def band(self):
        """Get the photometric filter/band used in the catalog."""
        return str(self._band)

    def copy(self):
        """Copy the current catalog to a new instance."""
        return copy.copy(self)

    def get_coordinates(self, obstime=None):
        """Get the skycoord positions from the catalog."""
        try:
            return self._coords.apply_space_motion(new_obstime=obstime)
        except ValueError:
            return copy.copy(self._coords)

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
        length = len(ra)
        ids = ['']*length
        nra = np.full(length, fill_value=np.nan, dtype='f8')
        ndec = np.full(length, fill_value=np.nan, dtype='f8')
        mags = np.full(length, fill_value=np.nan, dtype='f4')
        mags_error = np.full(length, fill_value=np.nan, dtype='f4')

        for i, v in enumerate(indexes):
            if v != -1:
                ids[i] = self._ids[v]
                nra[i] = cat_sk.ra.degree[v]
                ndec[i] = cat_sk.dec.degree[v]
                mags[i] = self._mags.nominal[v]
                mags_error[i] = self._mags.uncertainty[v]

        ncat = SourcesCatalog(nra, ndec, unit='degree', ids=ids,
                              mag=QFloat(mags, uncertainty=mags_error,
                                         unit=self._mags.unit))
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

        nc = copy.copy(self)
        nc._query = None
        nc._coords = self._coords[item]
        nc._ids = self._ids[item]
        if self._mags is not None:
            nc._mags = self._mags[item]
        return nc

    def __len__(self):
        return len(self._coords.ra.degree)


class _OnlineSourcesCatalog(SourcesCatalog, abc.ABC):
    """Sources Catalog based on online queries."""

    _query = None

    def __init__(self, center, radius, band=None):
        """Query the catalog and create the source catalog instance.

        Parameters
        ----------
        center: string, tuple or `astropy.coordinates.SkyCoord`
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
        band: string (optional)
            For catalogs with photometric information with multiple filters,
            the desired filter must be passed here.
            Default: None
        """
        self._center = astroquery_skycoord(center)
        self._radius = astroquery_radius(radius)
        if band is not None and band not in self.available_filters:
            raise ValueError(f'Filter {band} not available. Default '
                             f'filters are {self.available_filters}.')
        self._band = band

        # setup the catalog if needed
        self._setup_catalog()

        # perform the query
        logger.info('Quering region centered at %s with radius %s',
                    self._center, self._radius)
        logger.info('Using %s filter for photometry information.',
                    self._band)
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
