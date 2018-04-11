# Licensed under a 3-clause BSD style license - see LICENSE.rst
'''
Astrometry
----------

Default astrometry calibration functions.
'''
__all__ = ['wcs_xy2radec', 'wcs_radec2xy']


def wcs_xy2radec(x, y, wcs):
    """Convert x and y coordinates to RA and DEC using a WCS object."""
    return wcs.all_pix2world(x, y, 0.0, ra_dec_order=True)


def wcs_radec2xy(ra, dec, wcs):
    """Convert RA and DEC coordinates to x and y using a WCS object."""
    return wcs.all_world2pix(ra, dec, ra_dec_order=True)
