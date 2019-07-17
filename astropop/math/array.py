# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Small module for simple matrix works. Possible deprecated in future.
"""


import numpy as np


__all__ = ['xy2r']


def xy2r(x, y, data, xc, yc):
    r = np.hypot((x-xc), (y-yc))
    return np.ravel(r), np.ravel(data)
