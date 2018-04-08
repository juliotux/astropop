# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.table import Table

from ..astrometry import wcs_xy2radec
from ..py_utils import process_list, string_fix

__all__ = ['identify_stars']


def identify_stars(x, y, wcs, identify_catalog,
                   science_catalog=None, filter=None,
                   limit_angle='2 arcsec'):
    """Identify stars with coordinates x and y in a wcs frame and return
    pertinent parameters from the catalogs."""
    cat = identify_catalog
    ra, dec = wcs_xy2radec(x, y, wcs)

    name, mag, mag_err = cat.query_id_mag(ra, dec, filter,
                                          limit_angle=limit_angle)

    res = Table()
    if science_catalog is not None:
        sci = science_catalog
        sci_names, _, _ = sci.query_id_mag(ra, dec, None,
                                           limit_angle=limit_angle)
        res['sci_id'] = process_list(string_fix, sci_names)

    res['cat_id'] = process_list(string_fix, name)
    res['ra'] = ra
    res['dec'] = dec
    res['cat_mag'] = mag
    res['cat_mag_err'] = mag_err

    return res.as_array()
