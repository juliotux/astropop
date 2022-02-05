# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Data store containers with in-disk caching support and another features."""

from .framedata import FrameData
from .util import read_framedata, check_framedata
from .memmap import MemMapArray
from .compat import imhdus, EmptyDataError


__all__ = ['FrameData', 'read_framedata', 'check_framedata',
           'MemMapArray', 'imhdus', 'EmptyDataError']
