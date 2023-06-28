# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for loading data as FrameData."""

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


def read_framedata(obj, copy=False, **kwargs):
    """Read an object to a FrameData container.

    Parameters
    ----------
    obj: any compatible, see notes
      Object that will be readed to the FrameData.
    copy: bool (optional)
      If the object is already a FrameData, return a copy instead of the
      original one.
      Default: False

    Returns
    -------
    frame: `FrameData`
      The readed FrameData object.

    Notes
    -----
    - If obj is a string or `~pathlib.Path`, it will be interpreted as a file.
      File types will be checked. Just FITS format supported now.
    - If obj is `~astropy.io.fits.HDUList`, `~astropy.io.fits.HDUList` or
      `~astropy.nddata.CCDData`, they will be properly translated to
      `FrameData`.
    - If numbers or `~astropop.math.physical.QFloat`,
      `~astropy.units.Quantity`, they will be translated to a `FrameData`
      without metadata.
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
