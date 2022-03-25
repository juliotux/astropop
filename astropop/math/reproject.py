# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Module to handle spherical coordinates reprojection."""

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np


def gnomonic_tangential_projection(coords, center, pixel_scale=1./3600.):
    """Project a list of coordinates using gnomonic tangential projetion.

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord` or list
        List of coordinates to project. If a list is passed, it must be (n, 2)
        dimention containing decimal degrees RA and DEC coordinates.
    center : `~astropy.coordinates.SkyCoord` or list
        Coordinates of the center ofthe tangential plane. If a list, it must
        have length 2 and contain decimal degrees of RA and DEC coordinates.
    pixel_scale : float (optional)
        Angular size of each pixel (or unit) in the tangential plane.
        Default: 1./3600. (1 arcsec/px)

    Returns
    -------
    list :
        list containing (x, y) coordinates projected in the plane for each
        input position. (n, 2) dimension.
    """
    if isinstance(coords, (list, tuple, np.ndarray)):
        coords = SkyCoord(coords, unit='deg')
    if not isinstance(coords, SkyCoord):
        raise TypeError('coords must by SkyCoord compatible.')

    if not isinstance(center, SkyCoord):
        center = SkyCoord(*center, unit='deg')

    center = [center.ra.degree, center.dec.degree]

    w = WCS(naxis=2)
    w.wcs.crpix = [0, 0]
    w.wcs.crval = center
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cdelt = [pixel_scale, pixel_scale]

    return np.transpose(coords.to_pixel(w))
