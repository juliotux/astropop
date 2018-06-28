import numpy as np
from astropy.wcs import WCS

from ..astrometry.astrometrynet import solve_astrometry_xy
from ..catalogs import identify_stars


__all__ = ['identify_stars', 'solve_astrometry']


def solve_astrometry(table, header, shape, ra_key=None, dec_key=None,
                     plate_scale=None, use_previous_wcs=False):
    """Solves the astrometry of a field and return a valid wcs."""
    wcs = WCS(header, relax=True)
    if not wcs.wcs.ctype[0] or not use_previous_wcs:
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
