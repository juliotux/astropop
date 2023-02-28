# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Base classes for astronomical catalogs queries."""

import copy
import abc
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle

from ..math.physical import QFloat
from ..logger import logger
from ._online_tools import astroquery_radius, \
                           astroquery_skycoord


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
        magnitudes of the object in a `~astropop.math.QFloat` array. Photometry
        will be only available if this argument is set.
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
    query_table: `~astropy.table.Table` (optional)
        Optional table containing additional informations you may want to use
        for the sources. It must have the same number and order of the sources
        in the catalog.
    *args, **kwargs:
        Arguments to be passed to `~astropy.coordinates.SkyCoord`
        initialization. See `~astropy.coordinates.SkyCoord` docs for more
        details.
    """

    _base_table = None  # Store array of ids
    _mags_table = None  # Store magnitudes with filter colname in QFloat format
    _query = None  # Query table

    def __init__(self, *args, ids=None, mag=None,
                 query_table=None, **kwargs):
        self._base_table = Table()
        # initializate coords and skycoords using default kwargs.
        coords = SkyCoord(*args, **kwargs, copy=True)

        # IDs are stored in a numpy 1d-array
        if len(np.shape(ids)) != 1:
            raise ValueError('Sources ID must be a 1d array.')

        # initialize store table
        self._base_table['id'] = np.array(ids)
        self._base_table['coords'] = coords

        # Store base query table
        if query_table is not None:
            # TODO: check query_table length to match the sources
            self._query = query_table

        if mag is None:
            # simply pass if mag is None
            return

        # magnitudes are stored in a dict of QFloat
        if not isinstance(mag, dict):
            raise TypeError('mag must be a dictionary of magnitudes')
        for i in mag.keys():
            self._add_mags(i, mag[i])

    def _add_mags(self, band, mag):
        """Add a mag to the magnitude dict."""
        # initialize magnitude if not initializated.
        if self._mags_table is None:
            self._mags_table = Table()

        # TODO: if list of tuples (mag, err), split it
        # add magnitude to the dict
        m = QFloat(mag)
        if len(m) != len(self._base_table):
            raise ValueError('Lengths of magnitudes must be the same as '
                             'the number of sources.')
        self._mags_table[f'{band}'] = m.nominal
        self._mags_table[f'{band}_error'] = m.std_dev

    def _ensure_band(self, band):
        """Ensure there is a mags_table and band is in it."""
        if self._mags_table is None:
            raise ValueError('This SourcesCatalog has no photometic '
                             'information.')
        if band not in self.filters:
            raise ValueError(f'{band} not available.')

    def sources_id(self):
        """Get the list of sources id in catalog."""
        return self._base_table['id'].value

    def skycoord(self):
        """Get the sources coordinates in SkyCoord format."""
        return self.get_coordinates()

    def ra_dec_list(self):
        """Get the sources coordinates in [(ra, dec)] format."""
        sk = self.skycoord()
        return np.array(list(zip(sk.ra.degree, sk.dec.degree)))

    def ra(self):
        """Get the sources right ascension in degrees."""
        return np.array(self.skycoord().ra.degree)

    def dec(self):
        """Get the sources declination in degrees."""
        return np.array(self.skycoord().dec.degree)

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
        self._ensure_band(band)
        return QFloat(self._mags_table[f'{band}'],
                      self._mags_table[f'{band}_error'],
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
        mags = self.magnitude(band)
        return np.array(list(zip(mags.nominal, mags.std_dev)))

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

    def _extract_coords(self, ncat, nulls=None):
        """Extract coordinates in the form of a table.

        Nulls is a list of indexes to be set to a nan value.
        """
        base = ncat._base_table['coords']
        t = {}
        t['ra'] = base.ra
        t['dec'] = base.dec
        if nulls is not None and nulls != []:
            # nan coords for unmatched stars
            null_ra = np.nan*base.ra.unit
            null_dec = np.nan*base.dec.unit
            for i in nulls:
                t['ra'][i] = null_ra
                t['dec'][i] = null_dec
        try:
            t['pm_ra_cosdec'] = base.pm_ra_cosdec
            t['pm_dec'] = base.pm_dec
            if nulls is not None and nulls != []:
                for i in nulls:
                    # null pm for unmatched stards
                    t['pm_ra_cosdec'][i] = np.nan
                    t['pm_dec'][i] = np.nan
        except TypeError:
            pass

        return t

    def match_objects(self, ra, dec, limit_angle, obstime=None):
        """Find catalog objects matching the given coordinates.

        The matching is performed by getting the nearest catalog object from
        the ra, dec coordinate within the limit angle. No additional matching
        criteria is used.

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
        """
        # Match catalog to the objects using standard SkyCoord algorithm.
        obj_sk = SkyCoord(ra, dec, unit=['deg', 'deg'])
        cat_sk = self.get_coordinates(obstime)
        indx, dist, _ = obj_sk.match_to_catalog_sky(cat_sk)
        ncat = self.__getitem__(indx)

        # filtering
        nulls = [i for i, d in enumerate(dist) if d > Angle(limit_angle)]
        extra = {i: cat_sk.__getattribute__('_'+i)
                 for i in cat_sk._extra_frameattr_names}
        coords = self._extract_coords(ncat, nulls)
        for i in nulls:
            # null values for non-matched stars
            ncat._base_table['id'][i] = ''
            if ncat._mags_table is not None:
                ncat._mags_table[i] = [np.nan]*len(self._mags_table.colnames)
            if ncat._query is not None:
                ncat._query[i] = list(*np.zeros(1, dtype=ncat._query.dtype))
        ncat._base_table['coords'] = SkyCoord(**coords, **extra)
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

        nc = super(SourcesCatalog, self).__new__(SourcesCatalog)
        nc._base_table = Table(self._base_table[item])
        if self._mags_table is not None:
            nc._mags_table = Table(self._mags_table[item])
        if self._query is not None:
            nc._query = Table(self._query[item])
        return nc

    def __len__(self):
        return len(self._base_table)

    def table(self):
        """This catalog in a `~astropy.table.Table` instance."""
        t = Table()
        t['id'] = self._base_table['id']
        sk = self.skycoord()
        t['ra'] = sk.ra
        t['dec'] = sk.dec

        # only include pm informations if present
        try:
            t['pm_ra_cosdec'] = sk.pm_ra_cosdec
            t['pm_dec'] = sk.pm_dec
        except TypeError:
            pass

        if self._mags_table is not None:
            for i in self._mags_table.keys():
                t[i] = self._mags_table[i]
        return t

    @property
    def query_table(self):
        """The query table."""
        if self._query is not None:
            return Table(self._query)

    @property
    def query_colnames(self):
        """Get column names from query"""
        if self._query is not None:
            return copy.copy(self._query.colnames)

    @property
    def filters(self):
        """Get the list of all active available filters in the query."""
        if self._mags_table is not None:
            return [i for i in self._mags_table.keys() if '_error' not in i]


class _OnlineSourcesCatalog(SourcesCatalog, abc.ABC):
    """Sources Catalog based on online queries."""

    _query = None
    _available_filters = None

    def __init__(self, center, radius, band='all'):
        """Query the catalog and create the source catalog instance.

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
            Filters to query photometric informations. If None, photometric
            informations will be disabled. If ``'all'`` (default), all
            available filters will be queried. If a list, all filters in that
            list will be queried.

        Raises
        ------
        ValueError:
            If a ``band`` not available in the filters is passed.
        """
        self._center = astroquery_skycoord(center)
        self._radius = astroquery_radius(radius)
        self._filters = self._setup_filters(band)

        # setup the catalog if needed
        self._setup_catalog()

        # perform the query
        logger.info('Quering region centered at %s with radius %s',
                    self._center, self._radius)
        self._do_query()

    def _setup_filters(self, band):
        """Setup and check available filters."""
        if self._available_filters is None and band not in (None, 'all'):
            raise ValueError('No filters available for this catalog.')
        if band is None:
            return []
        if band == 'all':
            # [] for None available filters
            return copy.copy(self._available_filters) or []
        band = np.atleast_1d(band)
        for i in np.atleast_1d(band):
            if i not in self._available_filters:
                raise ValueError(f'Filter {i} not available for this catalog')
        return list(band)

    @abc.abstractmethod
    def _setup_catalog(self):
        """If a catalog setup is needed."""

    @abc.abstractmethod
    def _do_query(self):
        """Query the catalog. Must end with the catalog initialization."""

    @property
    def available_filters(self):
        """List available filters for the catalog."""
        return copy.copy(self._available_filters)

    @property
    def center(self):
        """Get the `~astropy.coordinates.SkyCoord` of the query center."""
        return SkyCoord(self._center)

    @property
    def radius(self):
        """Get the `~astropy.coordinates.Angle` of the earch angle limit."""
        return Angle(self._radius)

    @property
    def filters(self):
        """Get the list of all active available filters in the query."""
        return list(self._filters)
