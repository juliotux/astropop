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
    """Interpret paired RA/Dec values as numeric degrees or sexagesimal angles.

    Parameters
    ----------
    ra, dec
        Scalars or equal-length arrays. Values convertible to floating point
        are interpreted in degrees, including numeric strings. Otherwise try
        RA as hour angle and Dec as degrees with Astropy SkyCoord.
    skycoord
        If True (default), return SkyCoord; if False, return numerical RA/Dec
        in degrees as a pair of scalars or arrays.

    Returns
    -------
    coordinates
        Parsed sky coordinates in the selected representation.

    Raises
    ------
    ValueError
        One coordinate is scalar and the other an array, array lengths differ,
        or the supplied angles cannot be parsed as coordinates.

    Notes
    -----
    The helper guesses a format; it does not infer coordinate frame or epoch.
    Prefer an explicitly unit-bearing SkyCoord when the input format is known.

    Examples
    --------
    >>> from astropop.astrometry import guess_coordinates
    >>> position = guess_coordinates('08:00:00', '-30:00:00')
    >>> print(f'{position.ra.deg:.1f} {position.dec.deg:.1f}')
    120.0 -30.0
    """
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
