# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for loading data as FrameData."""

from __future__ import annotations

import typing as _t
from .._typing import ImageInput as _ImageInput

import os
import numpy as np
from astropy.io import fits
from astropy.units import Quantity
from astropy.nddata import CCDData

from .framedata import FrameData
from ._compat import _extract_ccddata, _extract_fits, imhdus


__all__ = ['check_framedata', 'read_framedata']


_fits_kwargs = ['hdu', 'unit', 'hdu_uncertainty',
                'hdu_mask', 'unit_key']


def read_framedata(obj: _ImageInput, copy: bool=False, **kwargs: _t.Any) -> FrameData:
    """Read an image into a FrameData container.

    Parameters
    ----------
    obj
        FrameData, FITS filename or path, image HDU, HDUList, CCDData,
        NumPy array, Quantity, or QFloat. Image data must be two-dimensional.
    copy
        If obj is already a FrameData, return a copy instead of the original.
        Default: False.
    **kwargs
        Constructor options for newly loaded frames. FITS input also accepts
        ``hdu``, ``unit``, ``hdu_uncertainty`` (default: ``UNCERT``),
        ``hdu_mask`` (default: ``MASK``), and ``unit_key`` (default: ``BUNIT``).
        For an existing FrameData with copy=True, options are passed to
        FrameData.copy; with copy=False, they are ignored.

    Returns
    -------
    frame
        Loaded image. Optional uncertainty and flag storage follows the
        current configuration when constructing or copying a frame.

    Notes
    -----
    Only FITS files are supported. The FITS reader currently selects the first
    HDU containing image data; pass an image HDU directly to select a specific
    image. Uncertainty extensions are interpreted as standard deviations.
    FITS and CCDData masks are imported as Boolean masks, not full pixel flags.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropop.framedata import read_framedata
    >>> frame = read_framedata(np.ones((2, 3)) * u.adu, dtype='float32')
    >>> print(frame.shape, frame.dtype.name, frame.unit)
    (2, 3) float32 adu

    >>> # Existing FrameData is returned directly unless a copy is requested.
    >>> print(read_framedata(frame) is frame)
    True
    >>> print(read_framedata(frame, copy=True) is frame)
    False
    """
    if isinstance(obj, FrameData):
        if copy:
            obj = obj.copy(**kwargs)
    elif isinstance(obj, CCDData):
        obj = FrameData(**_extract_ccddata(obj), **kwargs)
    elif isinstance(obj, (str, bytes, os.PathLike, fits.HDUList, *imhdus)):
        # separate kwargs to be sent to extractors
        fits_kwargs = {}
        for k in _fits_kwargs:
            if k in kwargs.keys():
                fits_kwargs[k] = kwargs.pop(k)
        obj = FrameData(**_extract_fits(obj, **fits_kwargs), **kwargs)
    elif isinstance(obj, Quantity):
        obj = FrameData(obj.value, unit=obj.unit, **kwargs)
    elif isinstance(obj, np.ndarray):
        obj = FrameData(obj, **kwargs)
    elif obj.__class__.__name__ == "QFloat":
        # if not do this, a cyclic dependency breaks the code.
        obj = FrameData(obj.nominal, unit=obj.unit,
                        uncertainty=obj.uncertainty, **kwargs)
    else:
        raise TypeError(f'Object {obj} is not compatible with FrameData.')

    return obj


check_framedata = read_framedata
