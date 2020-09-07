# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .framedata import FrameData  # noqa
from .memmap import MemMapArray, create_array_memmap, delete_array_memmap  # noqa
from .utils import check_framedata, framedata_read_fits, framedata_write_fits  # noqa
from .compat import imhdus, EmptyDataError  # noqa
