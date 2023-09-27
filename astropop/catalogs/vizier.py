# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in Vizier catalogs."""

from os import path, listdir
import numpy as np
import yaml
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astropy.time import Time

from ._sources_catalog import _OnlineSourcesCatalog, SourcesCatalog
from ._online_tools import astroquery_query
from ..py_utils import string_fix
from ..math import qfloat


__all__ = ['VizierSourcesCatalog', 'list_vizier_catalogs']


def _print_help(name, conf, available_filters):
    """Print the help for a catalog."""
    help = f"{name}: {conf['description']}\n"
    help += f"bibcode: {conf['bibcode']}\n\n"
    if available_filters:
        help += "Available filters are:\n"
        for i in conf['available_filters']:
            help += f"  - {i}: {conf['available_filters'][i]}\n"
    else:
        help += "This catalog has no photometric informations."
    return help


class VizierSourcesCatalog(_OnlineSourcesCatalog):
    """Sources catalog from Vizier plataform. See `help()` for details.

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
        self.name = path.splitext(path.basename(config_file))[0]
        with open(config_file, 'r') as f:
            self._conf = yaml.safe_load(f)
        self._table = self._conf['table']
        self._available_filters = self._get_available_filters()
        self._columns = self._conf['columns'] + self._get_mag_columns()

        self._setup_catalog()
        super().__init__(*args, **kwargs)

    def _get_available_filters(self):
        """Get the available filters."""
        filters = self._conf.get('available_filters', None)
        if filters is None:
            return
        return list(filters.keys())

    def _get_mag_columns(self):
        mag_conf = self._conf.get('magnitudes')
        mag_cols = []
        if mag_conf is not None:
            mag_key = mag_conf.get('mag_column', None)
            mag_cols.extend([mag_key.format(band=i)
                             for i in self._available_filters])
            err_mag_key = mag_conf.get('err_mag_column', None)
            if err_mag_key is not None:
                mag_cols.extend([err_mag_key.format(band=i)
                                 for i in self._available_filters])
        return mag_cols

    def _filter_magnitudes(self, query, band):
        """Get the qfloat magnitudes."""
        mag_key = self._conf['magnitudes']['mag_column'].format(band=band)
        err_mag_key = self._conf['magnitudes']['err_mag_column']
        err_mag_key = err_mag_key.format(band=band)

        unit = query[mag_key].unit
        mag = np.array(query[mag_key])
        mag_err = None
        if err_mag_key in query.colnames:
            err_unit = query[err_mag_key].unit
            mag_err = np.array(query[err_mag_key])
            mag_err = np.array([float(i) if i != '' else np.nan
                                for i in mag_err])
            if str(err_unit) == 'cmag':
                mag_err /= 100.0
        return qfloat(mag, uncertainty=mag_err, unit=unit)

    def _filter_coordinates(self, query, obstime, frame):
        """Get the SkyCoord coordinates."""
        rakey = self._conf['coordinates']['ra_column']
        deckey = self._conf['coordinates']['dec_column']
        ra = np.array(query[rakey])*query[rakey].unit
        dec = np.array(query[deckey])*query[deckey].unit
        if 'pm_ra_column' in self._conf['coordinates'] and \
           'pm_dec_column' in self._conf['coordinates']:
            pmrakey = self._conf['coordinates']['pm_ra_column']
            pmdeckey = self._conf['coordinates']['pm_dec_column']
            pmra = np.array(query[pmrakey])*query[pmrakey].unit
            pmdec = np.array(query[pmdeckey])*query[pmdeckey].unit
            return SkyCoord(ra, dec, frame=frame, obstime=obstime,
                            pm_ra_cosdec=pmra, pm_dec=pmdec)
        return SkyCoord(ra, dec, frame=frame, obstime=obstime)

    def _filter_ids(self, query):
        """Get the id names for the objects."""
        ids = self._conf['ids']
        if ids is None:
            return ['']*len(query)

        prepend = self._conf['ids'].get('prepend', None)
        id_key = self._conf['ids'].get('column', None)

        if not np.isscalar(id_key):
            sep = self._conf['ids'].get('separator', '-')
            ids = [(string_fix(j) for j in i) for i in query[id_key]]
            ids = [sep.join(list(i)) for i in ids]
        else:
            ids = [f'{string_fix(i)}' for i in query[id_key]]

        if prepend:
            ids = [f'{prepend} {string_fix(i)}' for i in ids]
        return ids

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

    def help(self):
        """Print the help for the catalog."""
        return _print_help(self.name, self._conf, self._available_filters)


def list_vizier_catalogs():
    root = path.join(path.dirname(__file__), 'vizier_catalogs')
    catalogs = 'Available pre-configured Vizier catalogs are:\n'
    for i in listdir(root):
        with open(path.join(root, i), 'r') as f:
            y = yaml.safe_load(f)
            catalogs += f'    `{i.replace(".yml", "")}`:'
            catalogs += f' :bibcode:`{y.get("bibcode", "")}`\n'
            catalogs += f'         {y.get("description", "")}\n'
    return catalogs


list_vizier_catalogs.__doc__ = "List available vizier catalogs\n\n"
list_vizier_catalogs.__doc__ += "Notes\n-----\n" + list_vizier_catalogs()


def __getattr__(name):
    filename = path.join(path.dirname(__file__), f'vizier_catalogs/{name}.yml')
    if path.exists(filename):
        class NewViz(VizierSourcesCatalog):
            def __init__(self, *args, **kwargs):
                super(NewViz, self).__init__(filename, *args, **kwargs)

        with open(filename, 'r') as f:
            conf = yaml.safe_load(f)
        available_filters = conf.get('available_filters', None)
        if available_filters:
            available_filters = list(available_filters.keys())

        # help function accessed before instance creation
        def help():
            return _print_help(name, conf, available_filters)

        NewViz.__doc__ = f"``{name}`` Vizier catalog."
        NewViz.help = staticmethod(help)
        NewViz.__doc__ += VizierSourcesCatalog.__doc__
        return NewViz
