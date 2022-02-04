# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query and match objects in Vizier catalogs."""

import copy
import numpy as np
from astroquery.vizier import Vizier, VizierClass

from .base_catalog import _BasePhotometryCatalog
from ._online_tools import astroquery_radius, \
                           astroquery_skycoord


__all__ = ['VizierCatalogClass', 'UCAC5Catalog',
           'UCAC4Catalog', 'GSC23Catalog', 'APASSCalatolg',
           'DENISCatalog', 'TWOMASSCatalog']


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
    _vizier = None

    def __init__(self, **kwargs):
        for i, v in kwargs.items():
            if i in self._valid_init_kwargs:
                self.__setattr__(i, v)
            else:
                raise ValueError('Invalid parameter {} passed to'
                                 ' VizierCatalogClass')

    @property
    def vizier(self):
        """Query operator instance."""
        if self._vizier is None:
            self._vizier = Vizier()
            self._vizier.ROW_LIMIT = -1
        return copy.copy(self._vizier)

    @vizier.setter
    def vizier(self, value):
        if not isinstance(value, VizierClass):
            raise ValueError(f'{value} is not a VizierClass instance.')
        self._vizier = value

    def _flux_keys(self, band):
        flux_key = self.flux_key.format(band=band)
        if self.flux_error_key is not None:
            flux_error_key = self.flux_error_key.format(band=band)
        else:
            flux_error_key = None
        return flux_key, flux_error_key

    def query_object(self, center):
        """Query a single object in the catalog."""
        center = astroquery_skycoord(center)
        return self._query(self.vizier.query_object,
                           center, catalog=self.vizier_table)[0]

    def query_region(self, center, radius):
        """Query all objects in a region."""
        radius = astroquery_radius(radius)
        center = astroquery_skycoord(center)
        viz = self.vizier
        return self._query(viz.query_region,
                           center, radius,
                           catalog=self.vizier_table)[0]

    def _id_resolve(self, idn):
        if self.prepend_id_key:
            if isinstance(self.prepend_id_key, (str, bytes)):
                id_key = self.prepend_id_key
            else:
                id_key = self.id_key
            idn = [f"{id_key} {i}" for i in idn]
            idn = np.array(idn)
        return idn

    def filter_flux(self, band, query=None):
        """Filter the flux data of a query."""
        self.check_filter(band)
        flux_key, flux_error_key = self._flux_keys(band)
        if query is None:
            query = self._last_query_table

        flux = np.array(query[flux_key].data)
        try:
            flux_error = np.array(query[flux_error_key].data)
        except KeyError:
            flux_error = np.array([np.nan]*len(flux))

        return flux, flux_error

    def match_objects(self, ra, dec, band=None, limit_angle='2 arcsec'):
        """Match objects from RA DEC list with this catalog."""
        flux_keys = ['flux', 'flux_error']
        table_props = [('id', ''), ('ra', np.nan), ('dec', np.nan),
                       ('flux', np.nan), ('flux_error', np.nan)]
        res = self._match_objects(ra, dec, band, limit_angle,
                                  flux_keys, table_props)

        return res


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

    def _flux_keys(self, band):
        if band in self.filt_conversion.keys():
            band = self.filt_conversion[band]
        return VizierCatalogClass._flux_keys(self, band)


GSC23Catalog = _GCS23Catalog()
