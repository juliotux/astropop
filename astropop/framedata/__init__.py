# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .framedata import FrameData  # noqa
from .util import read_framedata, check_framedata  # noqa
from .memmap import MemMapArray, create_array_memmap, delete_array_memmap  # noqa
from .compat import imhdus, EmptyDataError  # noqa
