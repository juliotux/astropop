# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in catalogs using TAP services."""

import copy
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.gaia import Gaia
from astropy.time import Time

from ._sources_catalog import _OnlineSourcesCatalog, SourcesCatalog
from ._online_tools import astroquery_query, string_fix
from ..math import QFloat


__all__ = ['GaiaDR3SourcesCatalog', 'gaiadr3']


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

    Raises
    ------
    ValueError:
        If a ``band`` not available in the filters is passed.
    """

    _available_filters = ['G', 'BP', 'RP']
    _columns = ['DESIGNATION', 'ref_epoch', 'ra', 'dec', 'pmra', 'pmdec',
                'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                'phot_g_mean_flux_over_error', 'phot_bp_mean_flux_over_error',
                'phot_rp_mean_flux_over_error', 'parallax', 'parallax_error',
                'radial_velocity', 'radial_velocity_error',
                'phot_variable_flag', 'non_single_star',
                'in_galaxy_candidates']

    def __init__(self, center, radius, band='all', max_g_mag=None):
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

    def _query_object_async(self, center, radius=None, columns=None):
        """Construct the query and launch job. Good for max_g_mag filtering."""
        # Based on astroquery.gaia.Gaia.query_object_async
        columns = ','.join(map(str, columns))

        ra = center.ra.degree
        dec = center.dec.degree

        if self._max_g_mag is not None:
            mag_filtering = f'AND phot_g_mean_mag < {self._max_g_mag}'
        else:
            mag_filtering = ''

        query = f"""
            SELECT
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
                ORDER BY
                  dist ASC
        """

        return self._g.launch_job_async(query,
                                        dump_to_file=False).get_results()

    def _do_query(self):
        self._query = astroquery_query(self._query_object_async,
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
