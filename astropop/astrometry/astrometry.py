# Licensed under a 3-clause BSD style license - see LICENSE.rst
'''
Astrometry
----------

Default astrometry calibration functions.
'''
import numpy as np
from astropy.wcs import WCS
from astropy.table import Table

from .astrometrynet import solve_astrometry_xy
from ..py_utils import process_list, string_fix


__all__ = ['solve_astrometry', 'identify_stars', 'wcs_xy2radec',
           'wcs_radec2xy']


def wcs_xy2radec(x, y, wcs):
    """Convert x and y coordinates to RA and DEC using a WCS object."""
    return wcs.all_pix2world(x, y, 0.0, ra_dec_order=True)


def wcs_radec2xy(ra, dec, wcs):
    """Convert RA and DEC coordinates to x and y using a WCS object."""
    return wcs.all_world2pix(ra, dec, ra_dec_order=True)


def solve_astrometry(table, header, shape, ra_key=None, dec_key=None,
                     plate_scale=None):
    """Solves the astrometry of a field and return a valid wcs."""
    wcs = WCS(header, relax=True)
    if not wcs.wcs.ctype[0]:
        im_params = {}
        if ra_key is not None and dec_key is not None:
            im_params['ra_key'] = ra_key
            im_params['dec_key'] = dec_key
        if plate_scale is not None:
            im_params['pltscl'] = plate_scale
            im_params['radius'] = 5*plate_scale*np.max(shape)/3600
        imw, imh = shape
        x, y = table['x'], table['y']
        flux = table['flux']
        wcs = solve_astrometry_xy(x, y, flux, header, imw, imh,
                                  image_params=im_params, return_wcs=True)
    return wcs
