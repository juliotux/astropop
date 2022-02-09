# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Small module for simple matrix works. Possible deprecated in future.
"""


import numpy as np


__all__ = ['xy2r', 'iraf_indices', 'trim_array', 'all_equal']


def xy2r(x, y, data, xc, yc):
    """Convert (x, y) values to distance of a (xc, yc) position."""
    r = np.hypot((x-xc), (y-yc))
    return np.ravel(r), np.ravel(data)


def trim_array(data, box_size, position, indices=None, origin=0):
    """Trim a numpy array around a position."""
    x, y = position
    # Correct for 1-based indexing
    x += origin
    y += origin

    dx = dy = float(box_size)/2

    x_min = max(int(x-dx), 0)
    x_max = min(int(x+dx)+1, data.shape[1])
    y_min = max(int(y-dy), 0)
    y_max = min(int(y+dy)+1, data.shape[0])

    d = data[y_min:y_max, x_min:x_max]

    if indices is None:
        return d, x-x_min, y-y_min

    xi = indices[1][y_min:y_max, x_min:x_max]
    yi = indices[0][y_min:y_max, x_min:x_max]
    return d, xi, yi


def all_equal(data):
    """Check if all elements of an array are equal."""
    return np.all(data == data.ravel()[0])


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
