# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import os
import pytest
import shutil
from urllib import request
from astroquery.skyview import SkyView
from astropy.coordinates import Angle, SkyCoord
from astropy.config import get_cache_dir
from astropy.io import fits
from astropy.nddata.ccddata import _generate_wcs_and_update_header
from astropy.wcs import WCS

from astropop.astrometry.astrometrynet import _solve_field, \
                                              solve_astrometry_image, \
                                              solve_astrometry_xy, \
                                              solve_astrometry_hdu, \
                                              AstrometrySolver
from astropop.astrometry.manual_wcs import wcs_from_coords
from astropop.astrometry.coords_utils import guess_coordinates

from numpy.testing import assert_array_equal, assert_array_almost_equal


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
    assert isinstance(nwcs, WCS)
    assert nwcs.naxis == 2
    # TODO: complete this test


def test_manual_wcs_top():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36 # arcsec/px
    ps_dev = ps/3600
    north = 'top'  # north to right
    wcs = wcs_from_coords(x, y, ra, dec, ps, north)
    assert_array_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_array_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(21, 11, 1), (9.9, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(1, 11, 1), (10.1, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(21, 21, 1), (9.9, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 21, 1), (10.1, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 1, 1), (10.1, -0.1))


def test_manual_wcs_left():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36 # arcsec/px
    ps_dev = ps/3600
    north = 'left'  # north to right
    wcs = wcs_from_coords(x, y, ra, dec, ps, north)
    assert_array_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_array_almost_equal(wcs.all_pix2world(11, 21, 1), (9.9, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(21, 11, 1), (10.0, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 11, 1), (10.0, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(11, 1, 1), (10.1, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(21, 21, 1), (9.9, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 21, 1), (9.9, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 1, 1), (10.1, 0.1))


def test_manual_wcs_bottom():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36 # arcsec/px
    ps_dev = ps/3600
    north = 'bottom'  # north to right
    wcs = wcs_from_coords(x, y, ra, dec, ps, north)
    assert_array_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_array_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(21, 11, 1), (10.1, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(1, 11, 1), (9.9, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(21, 21, 1), (10.1, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 21, 1), (9.9, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 1, 1), (9.9, 0.1))


def test_manual_wcs_right():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36 # arcsec/px
    ps_dev = ps/3600
    north = 'right'
    wcs = wcs_from_coords(x, y, ra, dec, ps, north)
    assert_array_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_array_almost_equal(wcs.all_pix2world(11, 21, 1), (10.1, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(21, 11, 1), (10.0, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 11, 1), (10.0, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(11, 1, 1), (9.9, 0.))
    assert_array_almost_equal(wcs.all_pix2world(21, 21, 1), (10.1, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 21, 1), (10.1, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 1, 1), (9.9, -0.1))


def test_manual_wcs_angle():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36 # arcsec/px
    ps_dev = ps/3600
    north = 45
    wcs = wcs_from_coords(x, y, ra, dec, ps, north)
    assert_array_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_array_almost_equal(wcs.all_pix2world(21, 21, 1), (10.0, 0.14142))
    assert_array_almost_equal(wcs.all_pix2world(1, 1, 1), (10.0, -0.14142))
    assert_array_almost_equal(wcs.all_pix2world(1, 21, 1), (10.14142, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(21, 1, 1), (9.858579, 0.0))


def test_manual_wcs_top_flip_ra():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36 # arcsec/px
    ps_dev = ps/3600
    north = 'top'
    flip = 'ra'
    wcs = wcs_from_coords(x, y, ra, dec, ps, north, flip=flip)
    assert_array_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_array_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(21, 11, 1), (10.1, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(1, 11, 1), (9.9, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(21, 21, 1), (10.1, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 21, 1), (9.9, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 1, 1), (9.9, -0.1))


def test_manual_wcs_top_flip_dec():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36 # arcsec/px
    ps_dev = ps/3600
    north = 'top'
    flip = 'dec'
    wcs = wcs_from_coords(x, y, ra, dec, ps, north, flip=flip)
    assert_array_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_array_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(21, 11, 1), (9.9, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(1, 11, 1), (10.1, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(21, 21, 1), (9.9, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 21, 1), (10.1, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 1, 1), (10.1, 0.1))


def test_manual_wcs_top_flip_all():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36 # arcsec/px
    ps_dev = ps/3600
    north = 'top'
    flip = 'all'
    wcs = wcs_from_coords(x, y, ra, dec, ps, north, flip=flip)
    assert_array_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(21, 11, 1), (10.1, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(1, 11, 1), (9.9, 0.0))
    assert_array_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, 0.1))
    assert_array_almost_equal(wcs.all_pix2world(21, 21, 1), (10.1, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 21, 1), (9.9, -0.1))
    assert_array_almost_equal(wcs.all_pix2world(1, 1, 1), (9.9, 0.1))


def test_raise_north_angle():
    with pytest.raises(ValueError) as exc:
        wcs_from_coords(0, 0, 0, 0, 0, 'not a direction')
        assert 'invalid value for north' in str(exc.value)


def test_guess_coords_float():
    ra = 10.0
    dec = 0.0
    assert_array_equal(guess_coordinates(ra, dec, skycoord=False), (ra, dec))

def test_guess_coords_strfloat():
    ra = "10.0"
    dec = "-27.0"
    assert_array_equal(guess_coordinates(ra, dec, skycoord=False), (10, -27))


def test_guess_coords_hexa_space():
    ra = "1 00 00"
    dec = "-43 30 00"
    assert_array_almost_equal(guess_coordinates(ra, dec, skycoord=False),
                              (15.0, -43.5))


def test_guess_coords_hexa_dots():
    ra = "1:00:00"
    dec = "-43:30:00"
    assert_array_almost_equal(guess_coordinates(ra, dec, skycoord=False),
                              (15.0, -43.5))


def test_guess_coords_skycord_float():
    ra = 10.0
    dec = 0.0
    sk = guess_coordinates(ra, dec, skycoord=True)
    assert isinstance(sk, SkyCoord)
    assert sk.ra.degree == ra
    assert sk.dec.degree == dec


def test_guess_coords_list_hexa():
    ra = ["1:00:00", "2:30:00"]
    dec = ["00:00:00", "1:00:00"]
    sra, sdec = guess_coordinates(ra, dec)
    assert isinstance(sk, SkyCoord)
    assert_array_almost_equal(sra, [15, 30.5])
    assert_array_almost_equal(sdec, [0, 1])


def test_guess_coords_list_float():
    ra = [10.0, 15, 20]
    dec = [0.0, 1.0, -1.0]
    sra, sdec = guess_coordinates(ra, dec)
    assert_array_equal(sra, ra)
    assert_array_equal(sdec, dec)


def test_guess_coords_list_diff():
    ra = np.arange(10)
    dec = np.arange(15)
    with pytest.raises(ValueError):
        guess_coordinates(ra, dec)


def test_guess_coords_list_nolist():
    ra = np.arange(10)
    dec = 1
    with pytest.raises(ValueError):
        guess_coordinates(ra, dec)
