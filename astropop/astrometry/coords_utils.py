# Licensed under a 3-clause BSD style license - see LICENSE.rst
'''Functions to easy handle with astronomical coordinates.'''

from __future__ import annotations

import numpy.typing as _npt

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units

__all__ = ['guess_coordinates']


def guess_coordinates(
    ra: _npt.ArrayLike,
    dec: _npt.ArrayLike,
    skycoord: bool=True,
) -> SkyCoord | tuple[float | _npt.NDArray, float | _npt.NDArray]:
    """Try to guess the format or ra and dec passed."""
    # TODO: this needs a refactor!
    # process as lists of coordinates
    if not np.isscalar(ra) and not np.isscalar(dec):
        if len(ra) != len(dec):
            raise ValueError('RA and Dec do not match in dimensions.')
        try:
            ra = np.array(ra, dtype='f8')
            dec = np.array(dec, dtype='f8')
            s = None
        except ValueError:
            s = SkyCoord(ra, dec, unit=(units.hourangle, units.deg))
    elif not np.isscalar(ra) or not np.isscalar(dec):
        raise ValueError('RA and Dec do not match in dimensions.')

    # process as single coordinates
    else:
        try:
            ra = float(ra)
            dec = float(dec)
            s = None
        except ValueError:
            # Assume (at least for now) that it's in sexagesimal
            s = SkyCoord(ra, dec, unit=(units.hourangle, units.deg))

    if s is not None:
        if skycoord:
            return s
        return s.ra.degree, s.dec.degree
    else:
        if skycoord:
            return SkyCoord(ra, dec, unit=(units.deg, units.deg))
        return ra, dec
