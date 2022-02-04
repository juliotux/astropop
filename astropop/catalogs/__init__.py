# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog managing and query."""

from .vizier import VizierCatalogClass, \
                    UCAC4Catalog, UCAC5Catalog, GSC23Catalog, \
                    APASSCalatolg, DENISCatalog, TWOMASSCatalog
from .simbad import SimbadCatalog, SimbadCatalogClass
from .local import TableCatalog, ASCIICatalog, FITSCatalog
from .utils import identify_stars


__all__ = ['identify_stars', 'catalogs_available',
           'VizierCatalogClass', 'SimbadCatalogClass', 'UCAC5Catalog',
           'SimbadCatalog', 'UCAC4Catalog', 'GSC23Catalog', 'APASSCalatolg',
           'DENISCatalog', 'TWOMASSCatalog', 'default_catalogs',
           'TableCatalog', 'ASCIICatalog', 'FITSCatalog']


default_catalogs = {'Simbad': SimbadCatalog,
                    'UCAC4': UCAC4Catalog,
                    'UCAC5': UCAC5Catalog,
                    'APASS': APASSCalatolg,
                    'DENIS': DENISCatalog,
                    '2MASS': TWOMASSCatalog,
                    'GSC2.3': GSC23Catalog}


catalogs_available = default_catalogs.keys()
