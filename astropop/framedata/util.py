# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for loading data as FrameData."""

import os
import numpy as np
import copy as cp
from astropy.io import fits
from astropy.nddata import CCDData
from astropy import units as u

from .framedata import FrameData
from .compat import _extract_ccddata, _extract_fits, imhdus
from .memmap import MemMapArray

__all__ = ['check_framedata', 'read_framedata']


def read_framedata(obj, copy=False, **kwargs):
    """Read an object to a FrameData container.

    Parameters
    ----------
    - obj: any compatible, see notes
      Object that will be readed to the FrameData.
    - copy: bool (optional)
      If the object is already a FrameData, return a copy instead of the
      original one.
      Default: False

    Returns
    -------
    - frame: `FrameData`
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
            obj = cp.deepcopy(obj)
    elif isinstance(obj, CCDData):
        obj = FrameData(**_extract_ccddata(obj, **kwargs))
    elif isinstance(obj, (str, bytes, os.PathLike)):
        obj = FrameData(**_extract_fits(obj, **kwargs))
    elif isinstance(obj, (fits.HDUList, *imhdus)):
        obj = FrameData(**_extract_fits(obj, **kwargs))
    elif isinstance(obj, (np.ndarray, MemMapArray)):
        obj = FrameData(obj)
    elif isinstance(obj, u.Quantity):
        obj = FrameData(obj.values, unit=obj.unit)
    elif obj.__class__.__name__ == "QFloat":
        # if not do this, a cyclic dependency breaks the code.
        obj = FrameData(obj.nominal, unit=obj.unit,
                        uncertainty=obj.uncertainty)
    else:
        raise ValueError(f'Object {obj} is not compatible with FrameData.')

    return obj


check_framedata = read_framedata
