# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Small utility to identify stars in a catalog."""

from astropy.table import Table

from ..py_utils import process_list, string_fix

__all__ = ['identify_stars']


def identify_stars(x, y, wcs, catalog, **kwargs):
    """Identify and name stars based on their x,y positions and WCS.

    Parameters
    ----------
    x, y: array_like
        x and y positions ofthe stars in the frame.
    wcs: `~astropy.wcs.WCS`
        World Coordinate System reference to identify the astronomical
        coordinates ofthe stars.
    catalog: `~astropop.catalogs._BaseCatalog`
        Catalog for star name matching.
    science_catalog: `~astropop.catalogs._BaseCatalog`
        Custom catalog for star name matching.
    band: `str` (optional)
        Photometric filter for flux estimation.
    limit_angle: `float`, `str` or `~astropy.coordinates.Angle` (optional)
        Maximum search limit to identify the star.
        Default: '2 arcsec'

    Returns
    -------
    `~astropy.table.Table`: All stars with id matched in the catalog.
    """
    limit_angle = kwargs.get('limit_angle', '2 arcsec')
    band = kwargs.get('band')
    cat = catalog
    ra, dec = wcs.all_pix2world(x, y, 1)  # Native wcs has 1 index counting

    cat_res = cat.match_objects(ra, dec, band, limit_angle=limit_angle)

    res = Table()
    res['ra'] = ra
    res['dec'] = dec
    res['cat_id'] = process_list(string_fix, cat_res['id'])
    res['cat_mag'] = cat_res['flux']
    res['cat_mag_err'] = cat_res['flux_error']

    if 'science_catalog' in kwargs.keys():
        sci = kwargs.get('science_catalog')
        sci_res = sci.match_objects(ra, dec, limit_angle=limit_angle)
        res['sci_id'] = process_list(string_fix, sci_res['id'])

    return res
