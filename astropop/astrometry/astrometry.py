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


def identify_stars(table, wcs, filter, identify_catalog,
                   science_catalog, identify_limit_angle='2 arcsec'):
    cat = identify_catalog
    x, y = table['x'], table['y']
    ra, dec = wcs_xy2radec(x, y, wcs)

    name, mag, mag_err = cat.query_id_mag(ra, dec, filter,
                                          limit_angle=identify_limit_angle)

    res = Table()
    if science_catalog is not None:
        sci = science_catalog
        limit_angle = identify_limit_angle
        sci_names, _, _ = sci.query_id_mag(ra, dec, None,
                                           limit_angle=limit_angle)
        res['sci_id'] = process_list(string_fix, sci_names)

    res['cat_id'] = process_list(string_fix, name)
    res['ra'] = ra
    res['dec'] = dec
    res['cat_mag'] = mag
    res['cat_mag_err'] = mag_err

    return res
