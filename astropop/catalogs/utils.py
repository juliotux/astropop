# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Small utility to identify stars in a catalog."""

from astropy.table import Table

from ..py_utils import process_list, string_fix

__all__ = ['identify_stars']


def identify_stars(x, y, wcs, identify_catalog,
                   science_catalog=None, filter=None,
                   limit_angle='2 arcsec'):
    """Identify stars with coordinates x and y in a wcs frame and return
    pertinent parameters from the catalogs."""
    cat = identify_catalog
    ra, dec = wcs.all_pix2world(x, y, 1)  # Native wcs has 1 index counting

    cat_res = cat.match_objects(ra, dec, filter, limit_angle=limit_angle)
    name = cat_res['id']
    mag = cat_res['flux']
    mag_err = cat_res['flux_error']

    res = Table()
    if science_catalog is not None:
        sci = science_catalog
        sci_res = sci.match_objects(ra, dec, limit_angle=limit_angle)
        res['sci_id'] = process_list(string_fix, sci_res['id'])

    res['cat_id'] = process_list(string_fix, name)
    res['ra'] = ra
    res['dec'] = dec
    res['cat_mag'] = mag
    res['cat_mag_err'] = mag_err

    return res.as_array()
