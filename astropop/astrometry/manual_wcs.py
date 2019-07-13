# Licensed under a 3-clause BSD style license - see LICENSE.rst

import six
import numpy as np
from astropy.wcs import WCS

from .coords_utils import guess_coordinates
from ..py_utils import check_iterable
from ..logger import logger

__all__ = ['wcs_from_coords']

_angles = {
    'left': 180,
    'right': 0,
    'top': 90,
    'bottom': 270
}


def wcs_from_coords(x, y, ra, dec, plate_scale, north, flip=None,
                    logger=logger):
    """Giving coordinates and plate scale, creates a WCS.
    x, y: float, pixel coordinates in image
    ra, dec: float, sky coordinates

    plate_scale in arcsec/pix

    north direction can be angles ('cw', from top axis) or
    ['right', 'top', 'left', 'bottom'] to angles [270, 0, 90, 180].

    flip means if some direction is mirrored in image.
    Can be 'ra', 'dec', 'all' or None
    The standard coordinates are ra and dec in the following order, ccw:
    W - N - E - S

    Problems:
    This algorith is not good for coordinates far from crpix. But is useful
    when you cannot solve with other algorithms. (Like just one star in field).
    """
    # FIXME: not flipping?
    sk = guess_coordinates(ra, dec)
    ra, dec = sk.ra.degree, sk.dec.degree

    if isinstance(north, six.string_types):
        if north in _angles.keys():
            north = _angles[north]
        else:
            raise ValueError('invalid value for north: {}'.format(north))

    # convert arcsec/pix to degree/pix
    if check_iterable(plate_scale):
        logger.warn("A list of plate scales given, using mean.")
        plate_scale = np.nanmean(plate_scale)
    plate_scale /= 3600

    # following the standard astrometry.net, all the delta informations
    # will be located inside the rotation/scale matrix pc
    deltra = -plate_scale if flip in ['ra', 'all'] else plate_scale
    deltde = -plate_scale if flip in ['dec', 'all'] else plate_scale

    rot = [[np.sin(np.radians(north)), -np.cos(np.radians(north))],
           [np.cos(np.radians(north)), np.sin(np.radians(north))]]

    pc = np.multiply(rot, [[deltra]*2, [deltde]*2])

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
