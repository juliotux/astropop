# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from ..framedata import MemMapArray


def _sep_fix_byte_order(data):
    """Fix the byte order of the data for SEP."""
    if isinstance(data, MemMapArray):
        data = np.array(data)

    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    if not data.dtype.isnative:
        data = data.byteswap().newbyteorder()
    if data.dtype.type in [np.uint, np.uintc, np.uint8, np.uint16, np.uint32,
                           np.int_, np.intc, np.int8, np.int16, np.int32]:
        data = data.astype(np.float_)
    return data
