# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Small module for simple matrix works. Possible deprecated in future.
"""

from __future__ import annotations

import numpy.typing as _npt


import numpy as np


__all__ = ['xy2r', 'iraf_indices', 'trim_array', 'all_equal']


def xy2r(
    x: _npt.ArrayLike,
    y: _npt.ArrayLike,
    data: _npt.ArrayLike,
    xc: float,
    yc: float,
) -> tuple[_npt.NDArray, _npt.NDArray]:
    """Convert (x, y) values to distance of a (xc, yc) position.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.math.array import xy2r
    >>> radius, values = xy2r(np.array([0, 3]), np.array([0, 4]),
    ...                      np.array([10, 20]), xc=0, yc=0)
    >>> print(radius.tolist(), values.tolist())
    [0.0, 5.0] [10, 20]
    """
    r = np.hypot((x-xc), (y-yc))
    return np.ravel(r), np.ravel(data)


def trim_array(
    data: _npt.NDArray,
    box_size: int | float,
    position: tuple[float, float],
    indices: tuple[_npt.NDArray, _npt.NDArray] | _npt.NDArray | None=None,
    origin: int=0,
) -> tuple[_npt.NDArray, float, float] | tuple[_npt.NDArray, _npt.NDArray, _npt.NDArray]:
    """Trim a numpy array around a position.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.math.array import trim_array
    >>> image = np.arange(25).reshape(5, 5)
    >>> cutout, cx, cy = trim_array(image, box_size=2, position=(2, 2))

    >>> # The returned center is relative to the cutout; upper bounds are inclusive.
    >>> print(cutout.shape, (cx, cy), int(cutout[cy, cx]))
    (3, 3) (1, 1) 12
    """
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


def all_equal(data: _npt.NDArray) -> np.bool_:
    """Check if all elements of an array are equal.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.math.array import all_equal
    >>> print(bool(all_equal(np.array([[2, 2], [2, 2]]))))
    True
    >>> print(bool(all_equal(np.array([2, 3]))))
    False
    """
    return np.all(data == data.ravel()[0])


def iraf_indices(
    data: _npt.NDArray,
) -> tuple[_npt.NDArray[np.int_], _npt.NDArray[np.int_]]:
    """Create (x, y) index arrays from a data matrix using IRAF convention.

    Iraf convention means (0, 0) on the center of bottom-left pixel.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.math.array import iraf_indices
    >>> x, y = iraf_indices(np.zeros((2, 3)))

    >>> # x follows columns and y follows rows, with zero-based pixel centers.
    >>> print(x.tolist(), y.tolist())
    [[0, 1, 2], [0, 1, 2]] [[0, 0, 0], [1, 1, 1]]
    """
    y, x = np.indices(data.shape)
    # Matches (0, 0) to the center of the pixel
    # FIXME: Check carefully if this is needed
    # x = x - 0.5
    # y = y - 0.5
    return x, y
