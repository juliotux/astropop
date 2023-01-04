# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in Vizier catalogs."""

from os import path
import numpy as np
import yaml
import functools
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astropy.time import Time

from ._sources_catalog import _OnlineSourcesCatalog, SourcesCatalog
from ._online_tools import astroquery_query
from ..py_utils import string_fix
from ..math import qfloat


__all__ = ['VizierSourcesCatalog']


class VizierSourcesCatalog(_OnlineSourcesCatalog):
    """Sources catalog from Vizier plataform.

    Parameters
    ----------
    config_file: string or dict
        Yaml configuration file containing the parameters to be used
        for catalog queries, like table id, column names, etc.
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

    def __init__(self, config_file, *args, **kwargs):
        with open(config_file, 'r') as f:
            self._conf = yaml.safe_load(f)
        self._table = self._conf['table']
        self._columns = self._conf['columns']
        self._available_filters = list(self._conf['available_filters'].keys())
        if self._conf['magnitudes'].get('mag_column'):
            mag_key = self._conf['magnitudes']['mag_column']
            for i in self._available_filters:
                self._columns.append(mag_key.format(band=i))
        if self._conf['magnitudes'].get('err_mag_column'):
            err_mag_key = self._conf['magnitudes']['err_mag_column']
            for i in self._available_filters:
                self._columns.append(err_mag_key.format(band=i))

        self._setup_catalog()
        super().__init__(*args, **kwargs)

    def _filter_magnitudes(self, query, band):
        """Get the qfloat magnitudes."""
        mag_key = self._conf['magnitudes']['mag_column'].format(band=band)
        err_mag_key = self._conf['magnitudes']['err_mag_column']
        err_mag_key = err_mag_key.format(band=band)

        unit = query[mag_key].unit
        mag = np.array(query[mag_key])
        if err_mag_key in query.colnames:
            err_unit = query[err_mag_key].unit
            mag_err = np.array(query[err_mag_key])
            mag_err = np.array([float(i) if i != '' else np.nan
                                for i in mag_err])
            if str(err_unit) == 'cmag':
                mag_err /= 100.0
        else:
            mag_err = None
        return qfloat(mag, uncertainty=mag_err, unit=unit)

    def _filter_coordinates(self, query, obstime, frame):
        """Get the SkyCoord coordinates."""
        rakey = self._conf['coordinates']['ra_column']
        deckey = self._conf['coordinates']['dec_column']
        ra = np.array(query[rakey])*query[rakey].unit
        dec = np.array(query[deckey])*query[deckey].unit
        if 'pm_ra_column' in query.colnames and \
           'pm_dec_column' in query.colnames:
            pmrakey = self._conf['coordinates']['pm_ra_column']
            pmdeckey = self._conf['coordinates']['pm_dec_column']
            pmra = np.array(query[pmrakey])*query[pmrakey].unit
            pmdec = np.array(query[pmdeckey])*query[pmdeckey].unit
            return SkyCoord(ra, dec, frame=frame, obstime=obstime,
                            pm_ra_cosdec=pmra, pm_dec=pmdec)
        else:
            return SkyCoord(ra, dec, frame=frame, obstime=obstime)

    def _filter_ids(self, query):
        """Get the id names for the objects."""
        ids = self._conf['ids']
        if ids is None:
            return ['']*len(query)

        prepend = self._conf['ids'].get('prepend', None)
        id_key = self._conf['ids'].get('column', None)

        if prepend:
            return [f'{prepend} {string_fix(i)}' for i in query[id_key]]
        else:
            return [f'{string_fix(i)}' for i in query[id_key]]

    def _filter_epoch(self, query):
        """Get the epoch for the coordinates."""
        ep = self._conf.get('epoch', None)
        if ep is None:
            # when no epoch information is available
            return
        if 'value' in ep:
            # fixed value
            return Time(ep['value'], format=ep['format'])
        if 'column' in ep:
            # value from
            column = np.atleast_1d(ep['column'])
            for i in column:
                if i in query.colnames:
                    return Time(query[i], format=ep['format'])

    def _setup_catalog(self):
        self._v = Vizier(catalog=self._table, columns=self._columns)
        self._v.ROW_LIMIT = -1

    def _do_query(self):
        q = astroquery_query(self._v.query_region,
                             self._center,
                             radius=self._radius)
        if len(q) == 0:
            raise RuntimeError('An error occured during online query.')
        self._query = q[0]
        ids = self._filter_ids(self._query)

        # perform magnitude filtering only if available
        mag = {}
        for filt in self.filters:
            mag[filt] = self._filter_magnitudes(self._query, filt)

        obstime = self._filter_epoch(self._query)
        frame = self._conf['coordinates'].get('frame', 'icrs')
        sk = self._filter_coordinates(self._query, obstime, frame)

        SourcesCatalog.__init__(self, sk, ids=ids, mag=mag)


def __getattr__(name):
    filename = path.join(path.dirname(__file__), f'vizier_catalogs/{name}.yml')
    if path.exists(filename):
        class NewViz(VizierSourcesCatalog):
            def __init__(self, *args, **kwargs):
                super(NewViz, self).__init__(filename, *args, **kwargs)
        return NewViz
