# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils for comparing and masking flags."""

from __future__ import annotations

import typing as _t
import numpy.typing as _npt

import numpy as np
from enum import Flag


__all__ = ['mask_from_flags']


def mask_from_flags(
    values: _npt.ArrayLike,
    flags_to_mask: Flag | _t.Sequence[Flag],
    allowed_flags_class: type[Flag] | tuple[type[Flag], ...] | None=None,
) -> _npt.NDArray[np.bool_]:
    """Generated a mask `~numpy.ndarray` from a given set of flags.

    Given an array or 2D image containing the flag values for each pixel or
    element, this function will generate a mask where all the pixels or
    elements that match any of the given flags will be masked.

    The comparison is performed using bitwise operations and the flags
    must be `~enum.IntFlag` instances and its values must be powers of 2.

    Parameters
    ----------
    values
        Array continaing the values to be evaluated.
    flags_to_mask
        List of flags to be masked.
    allowed_flags_class
        If not `None`, only flags from this class will be allowed. If `None`,
        any flag will be allowed.

    Returns
    -------
    mask
        Boolean mask with the same shape as ``values`` where all the pixels or
        elements that match any of the given flags will be masked.

    Examples
    --------
    >>> from astropop.flags import mask_from_flags
    >>> from astropop.framedata import PixelMaskFlags as F
    >>> values = [0, F.MASKED.value, (F.MASKED | F.SATURATED).value]

    >>> # Select saturation explicitly rather than treating every reason as a mask.
    >>> print(mask_from_flags(values, F.SATURATED).tolist())
    [False, False, True]
    """
    flag = 0
    if not isinstance(flags_to_mask, (list, tuple)):
        flags_to_mask = [flags_to_mask]

    for f in flags_to_mask:
        if not isinstance(f, Flag):
            raise TypeError('flags_to_mask must be a list of Flag')
        if allowed_flags_class is not None:
            if not isinstance(f, allowed_flags_class):
                raise TypeError('All flags must be from the same class')
        flag |= int(f.value)

    mask = np.greater(np.bitwise_and(values, flag), 0)

    return mask
