# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from photutils import CircularAperture
from astropop.photometry.background import background
from astropop.photometry.aperture import aperture_photometry, PhotometryFlags
from astropop.framedata import PixelMaskFlags
from astropop.photometry.aperture import _err_out_of_bounds, _err_pixel_flags
from astropop.testing import *

from .test_detection import gen_image


def test_photometry_flags_conformance_values():
    assert_equal(PhotometryFlags.REMOVED_PIXEL_IN_APERTURE.value, 1)
    assert_equal(PhotometryFlags.INTERPOLATED_PIXEL_IN_APERTURE.value, 2)
    assert_equal(PhotometryFlags.OUT_OF_BOUNDS.value, 4)
    assert_equal(PhotometryFlags.SATURATED_PIXEL_IN_APERTURE.value, 8)
    assert_equal(PhotometryFlags.REMOVED_PIXEL_IN_ANNULUS.value, 16)
    assert_equal(PhotometryFlags.INTERPOLATED_PIXEL_IN_ANNULUS.value, 32)
    assert_equal(PhotometryFlags.OUT_OF_BOUNDS_ANNULUS.value, 64)
    assert_equal(PhotometryFlags.NEARBY_SOURCES.value, 128)
    assert_equal(PhotometryFlags.NEARBY_SOURCES_ANNULUS.value, 256)
    assert_equal(PhotometryFlags.RECENTERING_FAILED.value, 512)


class TestApertureFlagsFunctions:
    def test_out_of_bound(self):
        f = PhotometryFlags.OUT_OF_BOUNDS
        assert_equal(_err_out_of_bounds((10, 10), [5], [5], 1, f)[0], 0)
        assert_equal(_err_out_of_bounds((10, 10), [10], [5], 1, f)[0], f.value)
        assert_equal(_err_out_of_bounds((10, 10), [5], [10], 1, f)[0], f.value)
        assert_equal(_err_out_of_bounds((10, 10), [-1], [5], 1, f)[0], f.value)
        assert_equal(_err_out_of_bounds((10, 10), [5], [-1], 1, f)[0], f.value)

    def test_pixel_flags(self):
        pf = PixelMaskFlags.MASKED
        f = PhotometryFlags.REMOVED_PIXEL_IN_APERTURE
        pixflags = np.zeros((10, 10), dtype=np.uint8)
        pixflags[5, 5] = pf.value
        ap = CircularAperture([(5, 5)], r=1)
        assert_equal(_err_pixel_flags(pixflags, ap, pf, f)[0], f.value)
        ap = CircularAperture([(1, 1)], r=1)
        assert_equal(_err_pixel_flags(pixflags, ap, pf, f)[0], 0)

    def test_pixel_flags_none(self):
        pf = PixelMaskFlags.MASKED
        f = PhotometryFlags.REMOVED_PIXEL_IN_APERTURE
        ap = CircularAperture([(5, 5)], r=1)
        assert_equal(_err_pixel_flags(None, ap, pf, f)[0], 0)


class TestAperturePhotometry:
    def test_single_star_manual(self):
        im = np.zeros((7, 7))
        im[3, 3] = 4
        im[[3, 3, 2, 4], [2, 4, 3, 3]] = 1
        phot = aperture_photometry(im, [3], [3], r=2, r_ann=None)
        assert_almost_equal(phot['flux'][0], 8, decimal=5)

    def test_single_star_manual_with_sky(self):
        im = np.ones((11, 11))
        im[5, 5] = 5
        im[[5, 5, 4, 6], [4, 6, 5, 5]] = 2
        phot = aperture_photometry(im, [5], [5], r=2, r_ann=(3, 5))
        assert_almost_equal(phot['flux'][0], 8, decimal=5)

    @pytest.mark.parametrize('flux', [10, 100, 1000, 10000, 100000])
    def test_single_star_gaussian_with_annulus(self, flux):
        im = gen_image((100, 100), [50], [50], flux=flux, sigma=2,
                       model='gaussian', rdnoise=0, sky=10, skip_poisson=True)
        phot = aperture_photometry(im, [50], [50], r=30, r_ann=(40, 50))
        assert_almost_equal(phot['flux'][0], flux, decimal=1-np.log10(flux))

    def test_multiple_star_gaussian_without_annulus(self):
        # use annulus None to not compute sky
        im = gen_image((100, 100), [20, 60], [20, 60], flux=[100, 100],
                       sigma=1, model='gaussian', rdnoise=0, sky=10,
                       skip_poisson=True)
        phot = aperture_photometry(im, [20, 60], [20, 60], r=5, r_ann=None)
        # expect include sky value
        expect = [10*np.pi*(5**2)+100]*2
        assert_almost_equal(phot['flux'], expect, decimal=0)

    def test_multiple_star_gaussian_with_annulus(self):
        im = gen_image((100, 100), [20, 60], [20, 60], flux=[100, 100],
                       sigma=1, model='gaussian', rdnoise=0, sky=10,
                       skip_poisson=True)
        phot = aperture_photometry(im, [20, 60], [20, 60], r=5, r_ann=(10, 20))
        assert_almost_equal(phot['flux'], [100, 100], decimal=0)

    def test_multiple_star_auto_r_ann(self):
        im = gen_image((100, 100), [20, 60], [20, 60], flux=[100, 100],
                       sigma=1, model='gaussian', rdnoise=0, sky=10,
                       skip_poisson=True)
        phot = aperture_photometry(im, [20, 60], [20, 60], r=5, r_ann='auto')
        assert_almost_equal(phot['flux'], [100, 100], decimal=0)
        # default is auto
        phot = aperture_photometry(im, [20, 60], [20, 60], r=5)
        assert_almost_equal(phot['flux'], [100, 100], decimal=0)

    def test_multiple_star_auto_r(self):
        im = gen_image((100, 100), [20, 60], [20, 60], flux=[100, 100],
                       sigma=1, model='gaussian', rdnoise=0, sky=10,
                       skip_poisson=True)
        phot = aperture_photometry(im, [20, 60], [20, 60], r='auto')
        # with optimal radius of 0.6371*fwhm, expect 82% of the flux
        assert_almost_equal(phot['flux'], [82, 82], decimal=0)
        phot = aperture_photometry(im, [20, 60], [20, 60])
        assert_almost_equal(phot['flux'], [82, 82], decimal=0)

    def test_recentering_single(self):
        im = gen_image((100, 100), [50.], [50.], flux=1000, sigma=2,
                       model='gaussian', rdnoise=0, sky=10, skip_poisson=True)
        phot = aperture_photometry(im, [52], [52], r=20, r_ann=(20, 30),
                                   recenter_limit=10,
                                   recenter_method='quadratic')
        assert_almost_equal(phot['x'][0], 50, decimal=1)
        assert_almost_equal(phot['y'][0], 50, decimal=1)
        assert_almost_equal(phot['original_x'][0], 52, decimal=1)
        assert_almost_equal(phot['original_y'][0], 52, decimal=1)
        assert_almost_equal(phot['flux'][0], 1000, decimal=-2)

    def test_recentering_multiple(self):
        im = gen_image((100, 100), [20, 60], [20, 60], flux=[100, 100],
                       sigma=1, model='gaussian', rdnoise=0, sky=10,
                       skip_poisson=True)
        phot = aperture_photometry(im, [22, 62], [22, 62], r=10, r_ann=(10, 20),
                                   recenter_limit=10,
                                   recenter_method='quadratic')
        assert_almost_equal(phot['x'], [20, 60], decimal=1)
        assert_almost_equal(phot['y'], [20, 60], decimal=1)
        assert_almost_equal(phot['original_x'], [22, 62], decimal=1)
        assert_almost_equal(phot['original_y'], [22, 62], decimal=1)
        assert_almost_equal(phot['flux'], [100, 100], decimal=0)

    def test_no_changes_inplace(self):
        # ensure aperture photometry is not changing the input image
        im = gen_image((1024, 1024), [50.], [50.], flux=1000, sigma=2,
                       model='gaussian', rdnoise=10, sky=1000,
                       skip_poisson=False)
        im2 = im.copy()
        bkg, rms = background(im, 64, 3)
        phot = aperture_photometry(im, [52], [52], r=20, r_ann=(20, 30),
                                   gain=1.5, bkg_error=rms)
        assert_almost_equal(im, im2, decimal=5)


class TestApertureFlags:
    @pytest.mark.parametrize('pos', [0, 11])
    def test_out_of_bounds_aperture(self, pos):
        im = np.ones((11, 11))
        phot = aperture_photometry(im, [5], [pos], r=2, r_ann=None)
        assert_true(phot['flags'][0] & PhotometryFlags.OUT_OF_BOUNDS.value)
        phot = aperture_photometry(im, [pos], [5], r=2, r_ann=None)
        assert_true(phot['flags'][0] & PhotometryFlags.OUT_OF_BOUNDS.value)

    @pytest.mark.parametrize('pos', [1, 10])
    def test_out_of_bounds_annulus(self, pos):
        im = np.ones((11, 11))
        phot = aperture_photometry(im, [5], [pos], r=1, r_ann=(5, 6))
        assert_true(phot['flags'][0] &
                    PhotometryFlags.OUT_OF_BOUNDS_ANNULUS.value)
        phot = aperture_photometry(im, [pos], [5], r=1, r_ann=(5, 6))
        assert_true(phot['flags'][0] &
                    PhotometryFlags.OUT_OF_BOUNDS_ANNULUS.value)

    def test_recentering_failed_single(self):
        im = gen_image((100, 100), [50.], [50.], flux=1000, sigma=2,
                       model='gaussian', rdnoise=0, sky=10, skip_poisson=True)
        phot = aperture_photometry(im, [45], [45], r=10, r_ann=None,
                                   recenter_limit=1.0,
                                   recenter_method='gaussian')
        assert_true(phot['flags'][0] & PhotometryFlags.RECENTERING_FAILED.value)

    def test_recentering_failed_multiple(self):
        im = gen_image((100, 100), [20, 60], [20, 60], flux=[100, 100],
                       sigma=1, model='gaussian', rdnoise=0, sky=10,
                       skip_poisson=True)
        phot = aperture_photometry(im, [20, 55], [20, 55], r=10, r_ann=None,
                                   recenter_limit=1.0,
                                   recenter_method='gaussian')
        assert_false(phot['flags'][0] & PhotometryFlags.RECENTERING_FAILED.value)
        assert_true(phot['flags'][1] & PhotometryFlags.RECENTERING_FAILED.value)
