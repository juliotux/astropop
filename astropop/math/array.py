# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Small module for simple matrix works. Possible deprecated in future.
"""


import numpy as np


__all__ = ['xy2r', 'iraf_indices']


def xy2r(x, y, data, xc, yc):
    """Convert (x, y) values to distance of a (xc, yc) position."""
    r = np.hypot((x-xc), (y-yc))
    return np.ravel(r), np.ravel(data)


def iraf_indices(data):
    """Create (x, y) index arrays from a data matrix using IRAF convention.

    Iraf convention means (0, 0) on the center of bottom-left pixel.
    """
    y, x = np.indices(data.shape)
    # Matches (0, 0) to the center of the pixel
    # FIXME: Check carefully if this is needed
    # x = x - 0.5
    # y = y - 0.5
    return x, y
