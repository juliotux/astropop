# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog managing and query."""

from .utils import identify_stars
from ._sources_catalog import SourcesCatalog
from .simbad import SimbadSourcesCatalog, simbad_query_id
from .vizier import UCAC4SourcesCatalog, APASS9SourcesCatalog, \
                    GSC242SourcesCatalog
from .tap import GaiaDR3SourcesCatalog


__all__ = ['identify_stars', 'SimbadSourcesCatalog', 'simbad_query_id',
           'SourcesCatalog', 'UCAC4SourcesCatalog', 'APASS9SourcesCatalog',
           'GSC242SourcesCatalog',
           'ucac4', 'apass9', 'gsc242', 'simbad']


ucac4 = UCAC4SourcesCatalog
apass9 = APASS9SourcesCatalog
gsc242 = GSC242SourcesCatalog
simbad = SimbadSourcesCatalog
