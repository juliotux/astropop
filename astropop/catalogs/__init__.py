# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog managing and query."""

from .online import default_catalogs, VizierCatalogClass, \
                    SimbadCatalogClass, SimbadCatalog, \
                    UCAC4Catalog, UCAC5Catalog, GSC23Catalog, \
                    APASSCalatolg, DENISCatalog, TWOMASSCatalog
from .local import TableCatalog, ASCIICatalog, FITSCatalog
from .utils import identify_stars


__all__ = ['identify_stars', 'catalogs_available',
           'VizierCatalogClass', 'SimbadCatalogClass', 'UCAC5Catalog',
           'SimbadCatalog', 'UCAC4Catalog', 'GSC23Catalog', 'APASSCalatolg',
           'DENISCatalog', 'TWOMASSCatalog', 'default_catalogs',
           'TableCatalog', 'ASCIICatalog', 'FITSCatalog']

catalogs_available = default_catalogs.keys()
