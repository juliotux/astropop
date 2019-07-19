# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import pytest
import shutil
from urllib import request
from astroquery.skyview import SkyView
from astropy.coordinates import Angle
from astropy.config import get_cache_dir
from astropy.io import fits
from astropy.nddata.ccddata import _generate_wcs_and_update_header

from astropop.astrometry.astrometrynet import _solve_field, \
                                              solve_astrometry_image, \
                                              solve_astrometry_xy, \
                                              solve_astrometry_hdu, \
                                              AstrometrySolver


def get_image_index():
    cache = get_cache_dir()
    ast_data = os.path.dirname(_solve_field)
    ast_data = os.path.dirname(ast_data)
    ast_data = os.path.join(ast_data, 'data')
    index = 'index-4107.fits'  # index-4202-28.fits'
    d = 'http://broiler.astrometry.net/~dstn/4100/' + index
    f = os.path.join(ast_data, index)
    if not os.path.isfile(f):
        request.urlretrieve(d, f)
    name = os.path.join(cache, 'm20_dss.fits')
    if not os.path.isfile(name):
        s = SkyView.get_images('M20', radius=Angle('60arcmin'),
                               pixels=2048, survey='DSS')
        s[0][0].writeto(name)
    return name, f


@pytest.mark.skipif('_solve_field is None')
def test_solve_astrometry_hdu(tmpdir):
    data, index = get_image_index()
    hdu = fits.open(data)[0]
    header, wcs = _generate_wcs_and_update_header(hdu.header)
    hdu.header = header
    nwcs = solve_astrometry_hdu(hdu, return_wcs=True)
    assert nwcs == wcs
