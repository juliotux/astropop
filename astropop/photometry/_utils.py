# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

def _sep_fix_byte_order(data):
    """Fix the byte order of the data for SEP."""
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    if not data.dtype.isnative:
        data = data.byteswap().newbyteorder()
    if data.dtype.type in [np.uint, np.uintc, np.uint8, np.uint16, np.uint32]:
        data = data.astype(np.intc)
    return data
