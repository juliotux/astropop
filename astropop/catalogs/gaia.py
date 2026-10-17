# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in catalogs using TAP services."""

import copy
import time
from contextlib import contextmanager
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.gaia import Gaia
from astropy.time import Time
from astropy.table import vstack

from ._sources_catalog import _OnlineSourcesCatalog, SourcesCatalog
from ._online_tools import astroquery_query, string_fix
from ..math import QFloat


__all__ = ['GaiaDR3SourcesCatalog', 'gaiadr3']


@contextmanager
def _gaia_query_timeout(client, timeout=300):
    """Limit socket waits using one budget for the whole Gaia query.

    The deadline is checked when a connection is created; its remaining time
    becomes that socket's inactivity timeout. This is not a hard wall-clock
    limit on DNS resolution, a continuously streaming response, or parsing.
    """
    # Gaia can accept an async job but leave it in EXECUTING for minutes, even
    # for a tiny result. A successful availability request does not rule this
    # out. Astroquery polls the job without an overall deadline, and its TAP
    # sockets have no explicit timeout, so even a status read can hang.
    #
    # The TAP client has no public timeout setter. Adapt the connection factory
    # only on this catalog's deepcopy of Gaia, preserving the configured host
    # and avoiding process-wide socket changes. Restore it on success or error.
    handler = client._TapPlus__getconnhandler()._TapConn__connectionHandler
    original = handler.get_connection
    deadline = time.monotonic() + timeout
    connections = []

    def get_connection(*args, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Gaia query timed out.')
        connection = original(*args, **kwargs)
        connection.timeout = remaining
        connections.append(connection)
        return connection

    handler.get_connection = get_connection
    try:
        yield
    finally:
        handler.get_connection = original
        for connection in connections:
            connection.close()


class GaiaDR3SourcesCatalog(_OnlineSourcesCatalog):
    """Sources catalog from Gaia-DR3 catalog.

    This class just wraps around `~astroquery.gaia.Gaia` class.

    Parameters
    ----------
    center: string, tuple or `~astropy.coordinates.SkyCoord`
        The center of the search field.
        If center is a string, can be an object name or the string
        containing the object coordinates. If it is a tuple, have to be
        (ra, dec) coordinates, in hexa or decimal degrees format.
    radius: string, float, `~astropy.coordinates.Angle`
        The radius to search. If None, the query will be performed as
        single object query mode. Else, the query will be performed as
        field mode. If a string value is passed, it must be readable by
        astropy.coordinates.Angle. If a float value is passed, it will
        be interpreted as a decimal degree radius.
    band: string or list(string) (optional)
        Filters to query photometric informations. If None, photometric
        informations will be disabled. If ``'all'`` (default), all
        available filters will be queried. If a list, all filters in that
        list will be queried. By default, all filters are available.
    max_g_mag: float (optional)
        Maximum G-band magnitude to query. If None, no magnitude
        filtering will be performed. Default is None.

    query_mode: {'async', 'sync'} (optional)
        Use asynchronous jobs (default) or synchronous requests. Synchronous
        queries fetch pages of at most 2000 sources and return the complete
        result sorted by distance, without creating asynchronous jobs.

    Raises
    ------
    ValueError:
        If a ``band`` not available in the filters is passed, or
        ``query_mode`` is not ``'async'`` or ``'sync'``.
    """

    _available_filters = ['G', 'BP', 'RP']
    _columns = ['DESIGNATION', 'ref_epoch', 'ra', 'dec', 'pmra', 'pmdec',
                'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                'phot_g_mean_flux_over_error', 'phot_bp_mean_flux_over_error',
                'phot_rp_mean_flux_over_error', 'parallax', 'parallax_error',
                'radial_velocity', 'radial_velocity_error',
                'phot_variable_flag', 'non_single_star',
                'in_galaxy_candidates']

    def __init__(self, center, radius, band='all', max_g_mag=None, *,
                 query_mode='async'):
        if query_mode not in ('async', 'sync'):
            raise ValueError("query_mode must be 'async' or 'sync'.")
        self._query_mode = query_mode
        self._setup_catalog()
        self._max_g_mag = max_g_mag
        _OnlineSourcesCatalog.__init__(self, center, radius=radius, band=band)

    def _setup_catalog(self):
        self._g = copy.deepcopy(Gaia)
        self._g.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
        self._g.ROW_LIMIT = -1

    def parallax(self):
        """Return the parallax for the sources."""
        return QFloat(self._query['parallax'],
                      self._query['parallax_error'],
                      unit='mas')

    def radial_velocity(self):
        """Return the radial velocity for the sources."""
        return QFloat(np.array(self._query['radial_velocity'], dtype='f4'),
                      np.array(self._query['radial_velocity_error'],
                               dtype='f4'),
                      unit='km/s')

    def phot_variable_flag(self):
        """Return the photometric variable flag for the sources."""
        return np.array(self._query['phot_variable_flag'] != 'CONSTANT')

    def non_single_star(self):
        """Return if each source is the non single star table."""
        return np.array(self._query['non_single_star'])

    def in_galaxy_candidates(self):
        """Return if the source is in galaxy candidates table."""
        return np.array(self._query['in_galaxy_candidates'])

    @staticmethod
    def _filter_magnitudes(query, band):
        """Get the qfloat magnitudes."""
        f = band.lower()
        mag = np.array(query[f'phot_{f}_mean_mag'])
        # Magnitude errors must be computed from SNR
        # sigma(mag) approx 1.1/snr
        mag_err = 1.1/np.array(query[f'phot_{f}_mean_flux_over_error'])
        return QFloat(mag, mag_err, unit='mag')

    def _query_object(self, center, radius=None, columns=None):
        """Query a cone using the selected TAP mode and sort by distance."""
        # Based on astroquery.gaia.Gaia.query_object_async. Sync mode bypasses
        # async job execution/polling, which was slow even when the same SQL
        # completed promptly through /sync. It can still suffer network or
        # server timeouts; it is an alternative path, not a timeout guarantee.
        sync = self._query_mode == 'sync'
        # Astroquery's synchronous launcher limits results to 2000 rows, even
        # with Gaia.ROW_LIMIT = -1. Page by unique source_id to avoid truncation.
        # Distance is not a safe cursor because multiple sources can tie.
        # Keep IDs as integers (Gaia IDs exceed float's exact-integer range),
        # and remove this extra column unless the caller requested it.
        include_id = sync and 'source_id' not in columns
        columns = ','.join(map(str, columns))
        if include_id:
            columns += ',source_id'
        row_limit = 'TOP 2000' if sync else ''
        order = 'source_id ASC' if sync else 'dist ASC'

        ra = center.ra.degree
        dec = center.dec.degree

        if self._max_g_mag is not None:
            mag_filtering = f'AND phot_g_mean_mag < {self._max_g_mag}'
        else:
            mag_filtering = ''

        query = f"""
            SELECT {row_limit}
                {columns},
                DISTANCE(
                    POINT('ICRS', {self._g.MAIN_GAIA_TABLE_RA},
                                  {self._g.MAIN_GAIA_TABLE_DEC}),
                    POINT('ICRS', {ra}, {dec})
                ) AS dist
                FROM
                  {self._g.MAIN_GAIA_TABLE}
                WHERE
                  1 = CONTAINS(POINT( 'ICRS', {self._g.MAIN_GAIA_TABLE_RA},
                                              {self._g.MAIN_GAIA_TABLE_DEC}),
                      CIRCLE('ICRS', {ra}, {dec}, {radius})
                  )
                  {mag_filtering}
                  {{page_filter}}
                ORDER BY
                  {order}
        """

        if not sync:
            return self._g.launch_job_async(query.format(page_filter=''),
                                            dump_to_file=False).get_results()

        pages = []
        last_id = None
        while True:
            page_filter = ('' if last_id is None else
                           f'AND source_id > {last_id}')
            page = self._g.launch_job(query.format(page_filter=page_filter),
                                      dump_to_file=False).get_results()
            if page is None:
                raise RuntimeError('No online catalog result found.')
            if len(page):
                next_id = int(page['source_id'][-1])
                if last_id is not None and next_id <= last_id:
                    raise RuntimeError('Gaia synchronous pagination stalled.')
                last_id = next_id
            pages.append(page)
            # A full page does not prove that the result is complete. An exact
            # multiple of 2000 therefore needs one final, empty request.
            if len(page) < 2000:
                break
        result = vstack(pages, join_type='exact') if len(pages) > 1 else pages[0]
        # Page order is by ID; the catalog API promises angular-distance order.
        result.sort('dist')
        if include_id:
            result.remove_column('source_id')
        return result

    def _do_query(self):
        # Keep the deadline outside astroquery_query: its timeout retries must
        # not each get a fresh five-minute budget. In sync mode this same budget
        # covers every page. An exception propagates instead of exposing a
        # partial catalog or silently switching back to the slow async path.
        with _gaia_query_timeout(self._g):
            self._query = astroquery_query(self._query_object,
                                           self._center,
                                           radius=self._radius.to(u.deg).value,
                                           columns=self._columns)
        sk = SkyCoord(self._query['ra'], self._query['dec'],
                      obstime=Time(self._query['ref_epoch'], format='jyear'),
                      pm_ra_cosdec=self._query['pmra'],
                      pm_dec=self._query['pmdec'])
        ids = np.array([string_fix(i) for i in self._query['DESIGNATION']])

        # perform magnitude filtering only if available
        mag = {}
        for f in self.filters:
            mag[f] = self._filter_magnitudes(self._query, f)

        SourcesCatalog.__init__(self, sk, ids=ids, mag=mag)


gaiadr3 = GaiaDR3SourcesCatalog
