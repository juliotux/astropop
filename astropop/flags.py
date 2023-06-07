# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils for comparing and masking flags."""

import numpy as np
from enum import Flag


__all__ = ['mask_from_flags']


def mask_from_flags(values, flags_to_mask, allowed_flags_class=None):
    """Generated a mask `~numpy.ndarray` from a given set of flags.

    Given an array or 2D image containing the flag values for each pixel or
    element, this function will generate a mask where all the pixels or
    elements that match any of the given flags will be masked.

    The comparison is performed using bitwise operations and the flags
    must be `IntFlag` instances and its values must be powers of 2.

    Parameters
    ----------
    values: `~numpy.ndarray`
        Array continaing the values to be evaluated.
    flags_to_mask : list of `Flag` or `Flag`
        List of flags to be masked.
    allowed_flags_class : `Flag` or `None`
        If not `None`, only flags from this class will be allowed. If `None`,
        any flag will be allowed.

    Returns
    -------
    mask : `~numpy.ndarray`
        Boolean mask with the same shape as `values` where all the pixels or
        elements that match any of the given flags will be masked.
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
