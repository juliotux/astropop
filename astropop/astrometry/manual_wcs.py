# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Create WCS manually."""

from __future__ import annotations

import typing as _t

import six
import numpy as np
from astropy.wcs import WCS

from .coords_utils import guess_coordinates


__all__ = ['wcs_from_coords']


_angles = {
    'left': 180,
    'right': 0,
    'top': 90,
    'bottom': 270
}


def wcs_from_coords(
    x: float,
    y: float,
    ra: str | float,
    dec: str | float,
    plate_scale: float,
    north: float | _t.Literal['left', 'right', 'top', 'bottom'],
    flip: _t.Literal['ra', 'dec', 'all'] | None=None,
) -> WCS:
    """Create an approximate tangent-plane WCS from one reference position.

    Parameters
    ----------
    x, y
        Reference pixel in FITS one-based coordinates, copied directly to
        CRPIX. Add one to zero-based detection or NumPy coordinates.
    ra, dec
        Sky position of the reference pixel. Numeric values are degrees;
        sexagesimal strings are parsed by ``guess_coordinates``.
    plate_scale
        Pixel scale in arcseconds per pixel.
    north
        Direction of increasing declination before applying ``flip``, measured
        counterclockwise from positive x with positive y displayed upward.
        Numeric angles are degrees. ``'right'`` is 0, ``'top'`` is 90,
        ``'left'`` is 180 and ``'bottom'`` is 270.
    flip
        ``'ra'`` reverses the RA direction, ``'dec'`` reverses declination,
        and ``'all'`` reverses both. None keeps the default orientation:
        with ``north='top'``, north is up and east is left.

    Returns
    -------
    wcs
        Two-dimensional TAN projection with the supplied reference pixel,
        reference sky position, scale and orientation.

    Notes
    -----
    This constructs a WCS from assumed geometry; it does not fit measured
    reference stars or model field distortion. Check the mapping against
    reference sources, particularly far from CRPIX. Astropy's high-level
    pixel methods use zero-based coordinates even though CRPIX is one-based.
    """
    sk = guess_coordinates(ra, dec)
    ra, dec = sk.ra.degree, sk.dec.degree

    if isinstance(north, six.string_types):
        if north in _angles.keys():
            north = _angles[north]
        else:
            raise ValueError(f'invalid value for north: {north}')

    # convert arcsec/pix to degree/pix
    plate_scale = float(plate_scale)  # ensure scalar
    plate_scale /= 3600

    # following the standard astrometry.net, all the delta informations
    # will be located inside the rotation/scale matrix pc
    deltra = plate_scale if flip == 'ra' else -plate_scale
    deltde = -plate_scale if flip == 'dec' else plate_scale
    if flip == 'all':
        north += 180

    rot = [[np.sin(np.radians(north)), -np.cos(np.radians(north))],
           [np.cos(np.radians(north)), np.sin(np.radians(north))]]

    pc = np.multiply(rot, [[deltra, deltra],
                           [deltde, deltde]])

    # and finally, we can construct the wcs
    w = WCS(naxis=2)
    w.wcs.crpix = [x, y]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [1, 1]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cunit = ['deg', 'deg']
    w.wcs.pc = pc
    # FIXME: Check why astrometry.net put pole in the crpix
    w.wcs.latpole = dec
    w.wcs.lonpole = 180

    return w
