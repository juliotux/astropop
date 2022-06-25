# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import numpy as np
import os
import pytest
from urllib import request
from astroquery.skyview import SkyView
from astropy.coordinates import Angle, SkyCoord
from astropy.config import get_cache_dir
from astropy.io import fits
from astropy.nddata.ccddata import _generate_wcs_and_update_header
from astropy.wcs import WCS

from astropop.astrometry.astrometrynet import _solve_field, fit_wcs, \
                                              solve_astrometry_image, \
                                              solve_astrometry_xy, \
                                              solve_astrometry_hdu
from astropop.astrometry.manual_wcs import wcs_from_coords
from astropop.astrometry.coords_utils import guess_coordinates
from astropop.photometry.aperture import aperture_photometry
from astropop.photometry.detection import starfind

from astropop.testing import *


def get_image_index():
    cache = get_cache_dir()
    ast_data = os.path.dirname(_solve_field)
    ast_data = os.path.dirname(ast_data)
    ast_data = os.path.join(ast_data, 'data')
    os.makedirs(ast_data, exist_ok=True)
    index = 'index-4107.fits'  # index-4202-28.fits'
    d = 'http://broiler.astrometry.net/~dstn/4100/' + index
    f = os.path.join(ast_data, index)
    if not os.path.isfile(f):
        request.urlretrieve(d, f)  # nosec
    name = os.path.join(cache, 'm20_dss.fits')
    if not os.path.isfile(name):
        s = SkyView.get_images('M20', radius=Angle('60arcmin'),
                               pixels=2048, survey='DSS')
        s[0][0].writeto(name)
    return name, f


def compare_wcs(wcs, nwcs):
    for i in [(100, 100), (1000, 1500), (357.5, 948.2), (2015.1, 403.7)]:
        res1 = np.array(wcs.all_pix2world(*i, 0))
        res2 = np.array(nwcs.all_pix2world(*i, 0))
        assert_almost_equal(res1, res2, decimal=3)


skip_astrometry = pytest.mark.skipif("_solve_field is None or "
                                     "os.getenv('SKIP_TEST_ASTROMETRY', "
                                     "False)")


@skip_astrometry
def test_solve_astrometry_hdu(tmpdir):
    data, index = get_image_index()
    hdu = fits.open(data)[0]
    header, wcs = _generate_wcs_and_update_header(hdu.header)
    hdu.header = header
    nwcs = solve_astrometry_hdu(hdu, return_wcs=True)
    assert_true(isinstance(nwcs, WCS))
    assert_equal(nwcs.naxis, 2)
    compare_wcs(wcs, nwcs)


@skip_astrometry
def test_solve_astrometry_xyl(tmpdir):
    data, index = get_image_index()
    hdu = fits.open(data)[0]
    header, wcs = _generate_wcs_and_update_header(hdu.header)
    hdu.header = header
    sources = starfind(hdu.data, 10, np.median(hdu.data),
                       np.std(hdu.data), 4)
    phot = aperture_photometry(hdu.data, sources['x'], sources['y'])
    imw, imh = hdu.data.shape
    nwcs = solve_astrometry_xy(phot['x'], phot['y'], phot['flux'], header,
                               imw, imh, return_wcs=True)
    assert_is_instance(nwcs, WCS)
    assert_equal(nwcs.naxis, 2)
    compare_wcs(wcs, nwcs)


@skip_astrometry
def test_solve_astrometry_image(tmpdir):
    data, index = get_image_index()
    hdu = fits.open(data)[0]
    header, wcs = _generate_wcs_and_update_header(hdu.header)
    hdu.header = header
    name = tmpdir.join('testimage.fits').strpath
    hdu.writeto(name)
    nwcs = solve_astrometry_image(name, return_wcs=True)
    assert_is_instance(nwcs, WCS)
    assert_equal(nwcs.naxis, 2)
    compare_wcs(wcs, nwcs)


@skip_astrometry
def test_fit_wcs(tmpdir):
    data, index = get_image_index()
    hdu = fits.open(data)[0]
    imw, imh = hdu.data.shape
    header, wcs = _generate_wcs_and_update_header(hdu.header)
    hdu.header = header
    sources = starfind(hdu.data, 10, np.median(hdu.data),
                       np.std(hdu.data), 4)
    sources['ra'], sources['dec'] = wcs.all_pix2world(sources['x'],
                                                      sources['y'], 1)
    nwcs = fit_wcs(sources['x'], sources['y'], sources['ra'], sources['dec'],
                   imw, imh)
    assert_is_instance(nwcs, WCS)
    assert_equal(nwcs.naxis, 2)
    compare_wcs(wcs, nwcs)


def test_manual_wcs_top():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36  # arcsec/px
    north = 'top'  # north to right
    wcs = wcs_from_coords(x, y, ra, dec, ps, north)
    assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, 0.1))
    assert_almost_equal(wcs.all_pix2world(21, 11, 1), (9.9, 0.0))
    assert_almost_equal(wcs.all_pix2world(1, 11, 1), (10.1, 0.0))
    assert_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, -0.1))
    assert_almost_equal(wcs.all_pix2world(21, 21, 1), (9.9, 0.1))
    assert_almost_equal(wcs.all_pix2world(1, 21, 1), (10.1, 0.1))
    assert_almost_equal(wcs.all_pix2world(1, 1, 1), (10.1, -0.1))


def test_manual_wcs_left():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36  # arcsec/px
    north = 'left'  # north to right
    wcs = wcs_from_coords(x, y, ra, dec, ps, north)
    assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_almost_equal(wcs.all_pix2world(11, 21, 1), (9.9, 0.0))
    assert_almost_equal(wcs.all_pix2world(21, 11, 1), (10.0, -0.1))
    assert_almost_equal(wcs.all_pix2world(1, 11, 1), (10.0, 0.1))
    assert_almost_equal(wcs.all_pix2world(11, 1, 1), (10.1, 0.0))
    assert_almost_equal(wcs.all_pix2world(21, 21, 1), (9.9, -0.1))
    assert_almost_equal(wcs.all_pix2world(1, 21, 1), (9.9, 0.1))
    assert_almost_equal(wcs.all_pix2world(1, 1, 1), (10.1, 0.1))


def test_manual_wcs_bottom():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36  # arcsec/px
    north = 'bottom'  # north to right
    wcs = wcs_from_coords(x, y, ra, dec, ps, north)
    assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, -0.1))
    assert_almost_equal(wcs.all_pix2world(21, 11, 1), (10.1, 0.0))
    assert_almost_equal(wcs.all_pix2world(1, 11, 1), (9.9, 0.0))
    assert_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, 0.1))
    assert_almost_equal(wcs.all_pix2world(21, 21, 1), (10.1, -0.1))
    assert_almost_equal(wcs.all_pix2world(1, 21, 1), (9.9, -0.1))
    assert_almost_equal(wcs.all_pix2world(1, 1, 1), (9.9, 0.1))


def test_manual_wcs_right():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36  # arcsec/px
    north = 'right'
    wcs = wcs_from_coords(x, y, ra, dec, ps, north)
    assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_almost_equal(wcs.all_pix2world(11, 21, 1), (10.1, 0.0))
    assert_almost_equal(wcs.all_pix2world(21, 11, 1), (10.0, 0.1))
    assert_almost_equal(wcs.all_pix2world(1, 11, 1), (10.0, -0.1))
    assert_almost_equal(wcs.all_pix2world(11, 1, 1), (9.9, 0.))
    assert_almost_equal(wcs.all_pix2world(21, 21, 1), (10.1, 0.1))
    assert_almost_equal(wcs.all_pix2world(1, 21, 1), (10.1, -0.1))
    assert_almost_equal(wcs.all_pix2world(1, 1, 1), (9.9, -0.1))


def test_manual_wcs_angle():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36  # arcsec/px
    north = 45
    wcs = wcs_from_coords(x, y, ra, dec, ps, north)
    assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_almost_equal(wcs.all_pix2world(21, 21, 1), (10.0, 0.14142))
    assert_almost_equal(wcs.all_pix2world(1, 1, 1), (10.0, -0.14142))
    assert_almost_equal(wcs.all_pix2world(1, 21, 1), (10.14142, 0.0))
    assert_almost_equal(wcs.all_pix2world(21, 1, 1), (9.858579, 0.0))


def test_manual_wcs_top_flip_ra():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36  # arcsec/px
    north = 'top'
    flip = 'ra'
    wcs = wcs_from_coords(x, y, ra, dec, ps, north, flip=flip)
    assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, 0.1))
    assert_almost_equal(wcs.all_pix2world(21, 11, 1), (10.1, 0.0))
    assert_almost_equal(wcs.all_pix2world(1, 11, 1), (9.9, 0.0))
    assert_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, -0.1))
    assert_almost_equal(wcs.all_pix2world(21, 21, 1), (10.1, 0.1))
    assert_almost_equal(wcs.all_pix2world(1, 21, 1), (9.9, 0.1))
    assert_almost_equal(wcs.all_pix2world(1, 1, 1), (9.9, -0.1))


def test_manual_wcs_top_flip_dec():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36  # arcsec/px
    north = 'top'
    flip = 'dec'
    wcs = wcs_from_coords(x, y, ra, dec, ps, north, flip=flip)
    assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
    assert_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, -0.1))
    assert_almost_equal(wcs.all_pix2world(21, 11, 1), (9.9, 0.0))
    assert_almost_equal(wcs.all_pix2world(1, 11, 1), (10.1, 0.0))
    assert_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, 0.1))
    assert_almost_equal(wcs.all_pix2world(21, 21, 1), (9.9, -0.1))
    assert_almost_equal(wcs.all_pix2world(1, 21, 1), (10.1, -0.1))
    assert_almost_equal(wcs.all_pix2world(1, 1, 1), (10.1, 0.1))


def test_manual_wcs_top_flip_all():
    # Checked with DS9
    x, y = (11, 11)
    ra, dec = (10.0, 0.0)
    ps = 36  # arcsec/px
    north = 'top'
    flip = 'all'
    wcs = wcs_from_coords(x, y, ra, dec, ps, north, flip=flip)
    assert_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, -0.1))
    assert_almost_equal(wcs.all_pix2world(21, 11, 1), (10.1, 0.0))
    assert_almost_equal(wcs.all_pix2world(1, 11, 1), (9.9, 0.0))
    assert_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, 0.1))
    assert_almost_equal(wcs.all_pix2world(21, 21, 1), (10.1, -0.1))
    assert_almost_equal(wcs.all_pix2world(1, 21, 1), (9.9, -0.1))
    assert_almost_equal(wcs.all_pix2world(1, 1, 1), (9.9, 0.1))


def test_raise_north_angle():
    with pytest.raises(ValueError) as exc:
        wcs_from_coords(0, 0, 0, 0, 0, 'not a direction')
        assert_in('invalid value for north', str(exc.value))


def test_guess_coords_float():
    ra = 10.0
    dec = 0.0
    assert_equal(guess_coordinates(ra, dec, skycoord=False), (ra, dec))


def test_guess_coords_strfloat():
    ra = "10.0"
    dec = "-27.0"
    assert_equal(guess_coordinates(ra, dec, skycoord=False), (10, -27))


def test_guess_coords_hexa_space():
    ra = "1 00 00"
    dec = "-43 30 00"
    assert_almost_equal(guess_coordinates(ra, dec, skycoord=False),
                              (15.0, -43.5))


def test_guess_coords_hexa_dots():
    ra = "1:00:00"
    dec = "-43:30:00"
    assert_almost_equal(guess_coordinates(ra, dec, skycoord=False),
                              (15.0, -43.5))


def test_guess_coords_skycord_float():
    ra = 10.0
    dec = 0.0
    sk = guess_coordinates(ra, dec, skycoord=True)
    assert_is_instance(sk, SkyCoord)
    assert_equal(sk.ra.degree, ra)
    assert_equal(sk.dec.degree, dec)


def test_guess_coords_skycord_hexa():
    ra = "1:00:00"
    dec = "00:00:00"
    sk = guess_coordinates(ra, dec, skycoord=True)
    assert_is_instance(sk, SkyCoord)
    assert_true(sk.ra.degree - 15 < 1e-8)
    assert_true(sk.dec.degree - 0 < 1e-8)


def test_guess_coords_list_hexa():
    ra = ["1:00:00", "2:30:00"]
    dec = ["00:00:00", "1:00:00"]
    sra, sdec = guess_coordinates(ra, dec, skycoord=False)
    assert_almost_equal(sra, [15, 37.5])
    assert_almost_equal(sdec, [0, 1])


def test_guess_coords_list_float():
    ra = [10.0, 15, 20]
    dec = [0.0, 1.0, -1.0]
    sra, sdec = guess_coordinates(ra, dec, skycoord=False)
    assert_equal(sra, ra)
    assert_equal(sdec, dec)


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
