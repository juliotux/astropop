# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from photutils import CircularAperture
from astropop.photometry.aperture import aperture_photometry, PhotometryFlags
from astropop.framedata import PixelMaskFlags
from astropop.photometry.aperture import _err_out_of_bounds, _err_pixel_flags
from astropop.testing import *


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
