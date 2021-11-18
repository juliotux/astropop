# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import sep
import numpy as np

from astropop.photometry import (background, sepfind, daofind, starfind,
                                 calc_fwhm, recenter_sources)
from astropop.photometry.detection import gen_filter_kernel
from astropop.testing import assert_almost_equal, assert_equal
from astropop.math.moffat import moffat_2d
from astropy.utils import NumpyRNGContext


def gen_bkg(size, level, rdnoise, rng_seed=123, dtype='f8'):
    """Generate a simple background image."""
    # create a level image
    im = np.ones(size, dtype)*level

    # reate the gaussian read noise image to sum
    with NumpyRNGContext(rng_seed):
        noise = np.random.normal(loc=0, scale=rdnoise, size=size)
    im += noise

    # poisonic not needed?
    return im


def gen_position_flux(size, number, low, high, rng_seed=123):
    """Generate x, y, and flux lists for stars."""
    for i in range(number):
        with NumpyRNGContext(rng_seed):
            x = np.random.randint(0, size[0], number)
        with NumpyRNGContext(rng_seed+i):
            y = np.random.randint(0, size[1], number)
            flux = np.random.randint(low, high, number)
    return x, y, flux


def gen_stars_moffat(size, x, y, flux, fwhm, rng_seed=123):
    """Generate stars image to add to background."""
    beta = 1.5
    alpha = fwhm/np.sqrt(2**(1/beta)-1)
    im = np.zeros(size)
    grid_y, grid_x = np.indices(size)
    for xi, yi, fi in zip(x, y, flux):
        im += moffat_2d(grid_x, grid_y, xi, yi, alpha, beta, fi, 0)

    return im


class Test_Detection_Conformance():
    # Conformance of secondary functions.
    def test_gen_kernel(self):
        assert_equal(gen_filter_kernel(3), [[0, 1, 0],
                                            [1, 4, 1],
                                            [0, 1, 0]])

        assert_equal(gen_filter_kernel(5), [[1, 2, 3, 2, 1],
                                            [2, 4, 6, 4, 2],
                                            [3, 6, 9, 6, 3],
                                            [2, 4, 6, 4, 2],
                                            [1, 2, 3, 2, 1]])

        assert_equal(gen_filter_kernel(7), [[1, 2, 3, 4, 3, 2, 1],
                                            [2, 4, 6, 8, 6, 4, 2],
                                            [3, 6, 9, 12, 9, 6, 3],
                                            [4, 8, 12, 16, 12, 8, 4],
                                            [3, 6, 9, 12, 9, 6, 3],
                                            [2, 4, 6, 8, 6, 4, 2],
                                            [1, 2, 3, 4, 3, 2, 1]])

    @pytest.mark.skip
    def test_sources_mask(self):
        pass


class Test_Background():
    def test_background_simple_nocosmic(self):
        size = (2048, 2048)
        level = 800
        rdnoise = 20
        image_test = gen_bkg(size, level, rdnoise)

        box_size = 64
        filter_size = 3
        global_bkg, global_rms = background(image_test, box_size, filter_size,
                                            mask=None, global_bkg=True)
        bkg, rms = background(image_test, box_size, filter_size,
                              mask=None, global_bkg=False)

        assert_equal(type(global_bkg), float)
        assert_equal(type(global_rms), float)
        assert_almost_equal(global_bkg, level, decimal=0)
        assert_almost_equal(global_rms, rdnoise, decimal=0)

        assert_equal(bkg.shape, size)
        assert_equal(rms.shape, size)
        assert_almost_equal(bkg, np.ones(size)*level, decimal=0)
        assert_almost_equal(rms, np.ones(size)*rdnoise, decimal=0)

    def test_background_simple_cosmic(self):
        size = (2048, 2048)
        level = 800
        rdnoise = 20
        image_test = gen_bkg(size, level, rdnoise)

        # add some cosmics
        for i in range(100):  # 100 single pixel cosmics
            x = np.random.randint(0, size[0]-1)
            y = np.random.randint(0, size[1]-1)
            image_test[x, y] = np.random.randint(16000, 64000)

        for i in range(4):  # 4 square block cosmics
            x = np.random.randint(0, size[0]-3)
            y = np.random.randint(0, size[1]-3)
            image_test[x:x+2, y:y+2] = np.random.randint(16000, 64000)

        box_size = 64
        filter_size = 3
        global_bkg, global_rms = background(image_test, box_size, filter_size,
                                            mask=None, global_bkg=True)
        bkg, rms = background(image_test, box_size, filter_size,
                              mask=None, global_bkg=False)

        assert_equal(type(global_bkg), float)
        assert_equal(type(global_rms), float)
        assert_almost_equal(global_bkg, level, decimal=0)
        assert_almost_equal(global_rms, rdnoise, decimal=0)

        assert_equal(bkg.shape, size)
        assert_equal(rms.shape, size)
        assert_almost_equal(bkg, np.ones(size)*level, decimal=0)
        assert_almost_equal(rms, np.ones(size)*rdnoise, decimal=0)

    def test_background_variablelevel_cosmics(self):
        size = (1024, 1024)
        y_i, x_i = np.indices(size)
        level = x_i*y_i/500 # level from 0 to 2000
        rdnoise = 20
        image_test = gen_bkg(size, level, rdnoise)

        # add some cosmics
        for i in range(100):  # 100 single pixel cosmics
            x = np.random.randint(0, size[0]-1)
            y = np.random.randint(0, size[1]-1)
            image_test[x, y] = np.random.randint(16000, 64000)

        for i in range(4):  # 4 square block cosmics
            x = np.random.randint(0, size[0]-3)
            y = np.random.randint(0, size[1]-3)
            image_test[x:x+2, y:y+2] = np.random.randint(16000, 64000)

        box_size = 64
        filter_size = 3
        global_bkg, global_rms = background(image_test, box_size, filter_size,
                                            mask=None, global_bkg=True)
        bkg, rms = background(image_test, box_size, filter_size,
                              mask=None, global_bkg=False)

        assert_equal(type(global_bkg), float)
        assert_equal(type(global_rms), float)
        assert_almost_equal(global_bkg, 389, decimal=0)
        assert_almost_equal(global_rms, 36, decimal=0)

        assert_equal(bkg.shape, size)
        assert_equal(rms.shape, size)
        # with stars, the dispersion increases
        assert_almost_equal(bkg, np.ones(size)*level, decimal=-1)
        # assert_almost_equal(rms, np.ones(size)*rdnoise, decimal=-1)

    def test_background_stars(self):
        size = (1024, 1024)
        stars_n = 50
        flux_low = 1500
        flux_high = 25000
        fwhm = 5
        level = 800
        rdnoise = 20
        x, y, f = gen_position_flux(size, stars_n, flux_low, flux_high)
        image_test = gen_bkg(size, level, rdnoise)
        image_test += gen_stars_moffat(size, x, y, f, fwhm)

        box_size = 64
        filter_size = 3
        global_bkg, global_rms = background(image_test, box_size, filter_size,
                                            mask=None, global_bkg=True)
        bkg, rms = background(image_test, box_size, filter_size,
                              mask=None, global_bkg=False)

        assert_equal(type(global_bkg), float)
        assert_equal(type(global_rms), float)
        assert_almost_equal(global_bkg, level, decimal=0)
        assert_almost_equal(global_rms, rdnoise, decimal=0)

        assert_equal(bkg.shape, size)
        assert_equal(rms.shape, size)
        # with stars, the dispersion increases
        assert_almost_equal(bkg, np.ones(size)*level, decimal=-1)
        assert_almost_equal(rms, np.ones(size)*rdnoise, decimal=-1)

    def test_background_stars_variablelevel(self):
        size = (1024, 1024)
        stars_n = 50
        flux_low = 1500
        flux_high = 25000
        fwhm = 5
        y_i, x_i = np.indices(size)
        level = x_i*y_i/500 # level from 0 to 2000
        rdnoise = 20
        x, y, f = gen_position_flux(size, stars_n, flux_low, flux_high)
        image_test = gen_bkg(size, level, rdnoise)
        image_test += gen_stars_moffat(size, x, y, f, fwhm)

        box_size = 64
        filter_size = 3
        global_bkg, global_rms = background(image_test, box_size, filter_size,
                                            mask=None, global_bkg=True)
        bkg, rms = background(image_test, box_size, filter_size,
                              mask=None, global_bkg=False)

        assert_equal(type(global_bkg), float)
        assert_equal(type(global_rms), float)
        assert_almost_equal(global_bkg, 389, decimal=0)
        assert_almost_equal(global_rms, 36, decimal=0)

        assert_equal(bkg.shape, size)
        assert_equal(rms.shape, size)
        # with stars, the dispersion increases
        assert_almost_equal(bkg, np.ones(size)*level, decimal=-1)
        # here the rms goes bigger with the level
        # assert_almost_equal(rms, np.ones(size)*rdnoise, decimal=-1)

@pytest.mark.skip
class Test_SEP_Detection():
    # segmentation detection. Must detect all shapes of sources
    def test_sepfind_one_star(self):
        pass

    def test_sepfind_strong_and_weak(self):
        pass

    def test_sepfind_four_stars_fixed_position(self):
        pass

    def test_sepfind_multiple_stars(self):
        pass

    def test_sepfind_segmentation_map(self):
        pass

@pytest.mark.skip
class Test_DAOFind_Detection():
    # DAOFind detection. Only round unsturaded stars
    def test_daofind_one_star(self):
        pass

    def test_daofind_strong_and_weak(self):
        pass

    def test_daofind_four_stars_fixed_position(self):
        pass

    def test_daofind_multiple_stars(self):
        pass

    def test_daofind_reject_sharpness_roundness(self):
        pass

@pytest.mark.skip
class Test_StarFind():
    # Our combined iterative method
    def test_starfind_calc_fwhm(self):
        pass

    def test_starfind_recenter_sources(self):
        pass

    def test_starfind_one_star(self):
        pass

    def test_starfind_strong_weak(self):
        pass

    def test_starfind_blind_fwhm(self):
        pass

    def test_starfind_rejection(self):
        pass