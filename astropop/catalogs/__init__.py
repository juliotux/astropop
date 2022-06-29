# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog managing and query."""

from .utils import identify_stars
from ._sources_catalog import SourcesCatalog
from .simbad import SimbadSourcesCatalog, simbad_query_id
from .vizier import UCAC4SourcesCatalog, APASS9SourcesCatalog


__all__ = ['identify_stars', 'SimbadSourcesCatalog', 'simbad_query_id',
           'SourcesCatalog', 'UCAC4SourcesCatalog', 'APASS9SourcesCatalog']
