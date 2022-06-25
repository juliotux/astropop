# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np

from astropop.photometry import (background, sepfind, daofind, starfind,
                                 calc_fwhm, recenter_sources)
from astropop.photometry.detection import gen_filter_kernel, DAOFind, \
                                          _sep_fix_byte_order
from astropop.framedata import MemMapArray
from astropop.math.moffat import moffat_2d
from astropop.math.gaussian import gaussian_2d
from astropop.math.array import trim_array
from astropy.utils import NumpyRNGContext
from astropy.stats import gaussian_fwhm_to_sigma

from astropop.testing import *

def gen_bkg(size, level, rdnoise, rng_seed=123, dtype='f8'):
    """Generate a simple background image."""
    # create a level image
    im = np.ones(size[::-1], dtype)*level

    # reate the gaussian read noise image to sum
    with NumpyRNGContext(rng_seed):
        noise = np.random.normal(loc=0, scale=rdnoise, size=size[::-1])
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


def gen_stars_moffat(size, x, y, flux, fwhm):
    """Generate stars image to add to background."""
    beta = 1.5
    alpha = fwhm/np.sqrt(2**(1/beta)-1)
    im = np.zeros(size[::-1])
    grid_y, grid_x = np.indices(size[::-1])
    for xi, yi, fi in zip(x, y, flux):
        imi, gxi, gyi = trim_array(np.zeros_like(im), box_size=5*fwhm,
                                   position=(xi, yi),
                                   indices=(grid_y, grid_x))
        imi += moffat_2d(gxi, gyi, xi, yi, alpha, beta, fi, 0)
        im[gyi.min():gyi.max()+1, gxi.min():gxi.max()+1] += imi

    return im


def gen_stars_gaussian(size, x, y, flux, sigma, theta):
    """Generate stars image to add to background."""
    im = np.zeros(size[::-1])
    grid_y, grid_x = np.indices(size[::-1])

    try:
        sigma_x, sigma_y = sigma
    except (TypeError, ValueError):
        sigma_x = sigma_y = sigma

    bc = np.broadcast(x, y, flux, sigma_x, sigma_y, theta)

    for xi, yi, fi, sxi, syi, ti in zip(*bc.iters):
        imi, gxi, gyi = trim_array(np.zeros_like(im), box_size=10*sxi,
                                   position=(xi, yi),
                                   indices=(grid_y, grid_x))
        imi += gaussian_2d(gxi, gyi, xi, yi, sxi, syi, ti, fi, 0)
        im[gyi.min():gyi.max()+1, gxi.min():gxi.max()+1] += imi

    return im


def gen_image(size, x, y, flux, sky, rdnoise, model='gaussian', **kwargs):
    """Generate a full image of stars with noise."""
    im = gen_bkg(size, sky, rdnoise)

    if model == 'moffat':
        fwhm = kwargs.pop('fwhm')
        im += gen_stars_moffat(size, x, y, flux, fwhm)
    if model == 'gaussian':
        sigma = kwargs.pop('sigma', 2.0)
        theta = kwargs.pop('theta', 0)
        im += gen_stars_gaussian(size, x, y, flux, sigma, theta)

    # can pass the poisson noise
    if not kwargs.get('skip_poisson', False):
        # prevent negative number error
        negatives = np.where(im < 0)
        im = np.random.poisson(np.absolute(im))
        # restore the negatives
        im[negatives] = -im[negatives]
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


class Test_SepFixByteOrder():
    def test_sep_fix_byte_order_memmaparray(self):
        arr = MemMapArray(np.arange(10).reshape(2, 5))
        arr1 = _sep_fix_byte_order(arr)
        assert_equal(arr1, arr)
        assert_is_instance(arr1, np.ndarray)
        assert_true(arr1.dtype.isnative)
        assert_true(arr1.flags['C_CONTIGUOUS'])
        assert_equal(arr1.dtype, np.dtype('float64'))

    def test_sep_fix_byte_order_f4(self):
        arr = np.arange(10, dtype='f4').reshape(2, 5)
        arr1 = _sep_fix_byte_order(arr)
        assert_equal(arr1, arr)
        assert_is_instance(arr1, np.ndarray)
        assert_true(arr1.dtype.isnative)
        assert_true(arr1.flags['C_CONTIGUOUS'])
        assert_equal(arr1.dtype, np.dtype('float32'))

    @pytest.mark.parametrize('dtype', ['i1', 'i2', 'i4', 'i8',
                                       'u1', 'u2', 'u4', 'u8'])
    def test_sep_fix_byte_order_dtypes(self, dtype):
        arr = np.arange(10, dtype=dtype).reshape(2, 5)
        arr1 = _sep_fix_byte_order(arr)
        assert_equal(arr1, arr)
        assert_is_instance(arr1, np.ndarray)
        assert_true(arr1.dtype.isnative)
        assert_true(arr1.flags['C_CONTIGUOUS'])
        assert_equal(arr1.dtype, np.dtype('float64'))

    @pytest.mark.parametrize('byteorder', ['>', '<'])
    @pytest.mark.parametrize('dtype', ['i1', 'i2', 'i4', 'i8',
                                       'u1', 'u2', 'u4', 'u8',
                                       'f8'])
    def test_sep_fix_byte_order_byteorder(self, byteorder, dtype):
        dtype = f"{byteorder}{dtype}"
        arr = np.arange(10, dtype=dtype).reshape(2, 5)
        arr1 = _sep_fix_byte_order(arr)
        assert_equal(arr1, arr)
        assert_is_instance(arr1, np.ndarray)
        assert_true(arr1.dtype.isnative)
        assert_true(arr1.flags['C_CONTIGUOUS'])
        assert_equal(arr1.dtype, np.dtype('float64'))

    def test_sep_fix_byte_order_contiguous(self):
        arr = np.arange(10, dtype='f4').reshape(2, 5).T
        arr1 = _sep_fix_byte_order(arr)
        assert_equal(arr1, arr)
        assert_is_instance(arr1, np.ndarray)
        assert_true(arr1.dtype.isnative)
        assert_true(arr1.flags['C_CONTIGUOUS'])
        assert_equal(arr1.dtype, np.dtype('float32'))


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

    def test_background_simple_negative_sky(self):
        size = (2048, 2048)
        level = -100
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
        level = x_i*y_i/500  # level from 0 to 2000
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
        level = x_i*y_i/500  # level from 0 to 2000
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


@pytest.mark.flaky(max_runs=10, min_passes=1)
class Test_SEP_Detection():
    # segmentation detection. Must detect all shapes of sources

    def resort_sources(self, x, y, f):
        """Random sources are random. We must resort to compare."""
        # For SEP, resort using x order
        order = np.argsort(y)
        return x[order], y[order], f[order]

    def test_sepfind_one_star(self):
        size = (128, 128)
        pos = (64, 64)
        sky = 20
        sky = 800
        rdnoise = 20
        flux = 32000
        sigma = 3
        theta = 0
        threshold = 10
        im = gen_image(size, [pos[0]], [pos[1]], [flux], sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = sepfind(im, threshold, sky, rdnoise)

        assert_equal(len(sources), 1)
        assert_almost_equal(sources['x'][0], 64, decimal=0)
        assert_almost_equal(sources['y'][0], 64, decimal=0)

    def test_sepfind_negative_sky(self):
        size = (128, 128)
        pos = (64, 64)
        sky = 0
        rdnoise = 20
        flux = 32000
        sigma = 3
        theta = 0
        threshold = 10

        im = gen_image(size, [pos[0]], [pos[1]], [flux], sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = sepfind(im, threshold, sky, rdnoise)

        assert_equal(len(sources), 1)
        assert_almost_equal(sources['x'][0], 64, decimal=0)
        assert_almost_equal(sources['y'][0], 64, decimal=0)

    def test_sepfind_strong_and_weak(self):
        size = (128, 128)
        posx = (60, 90)
        posy = (20, 90)
        sky = 800
        rdnoise = 20
        flux = (32000, 600)
        sigma = 1.5
        theta = 0
        im = gen_image(size, posx, posy, flux, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = sepfind(im, 3, sky, rdnoise)

        assert_equal(len(sources), 2)
        assert_almost_equal(sources['x'], posx, decimal=0)
        assert_almost_equal(sources['y'], posy, decimal=0)

    def test_sepfind_four_stars_fixed_position(self):
        size = (1024, 1024)
        posx = (10, 120, 500, 1000)
        posy = (20, 200, 600, 800)
        sky = 800
        rdnoise = 20
        flux = (15000, 1500, 1500, 35000)
        sigma = 3
        theta = 0
        im = gen_image(size, posx, posy, flux, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = sepfind(im, 3, sky, rdnoise)

        assert_equal(len(sources), 4)
        assert_almost_equal(sources['x'], posx, decimal=0)
        assert_almost_equal(sources['y'], posy, decimal=0)

    def test_sepfind_multiple_stars(self):
        size = (1024, 1024)
        number = 15
        low = 2000
        high = 30000
        sky = 800
        rdnoise = 20
        sigma = 3
        theta = 0

        x, y, f = gen_position_flux(size, number, low, high, rng_seed=456)
        im = gen_image(size, x, y, f, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = sepfind(im, 5, sky, rdnoise)

        x, y, f = self.resort_sources(x, y, f)

        assert_equal(len(sources), number)
        assert_almost_equal(sources['x'], x, decimal=0)
        assert_almost_equal(sources['y'], y, decimal=0)

    def test_sepfind_one_star_subpixel(self):
        size = (128, 128)
        pos = (54.32, 47.86)

        im = gen_image(size, [pos[0]], [pos[1]], [45000], 800, 0,
                       sigma=[3], skip_poisson=True)
        sources = sepfind(im, 5, 800, 10)
        assert_equal(len(sources), 1)
        # no error, 2 decimals ok!
        assert_almost_equal(sources[0]['x'], pos[0], decimal=2)
        assert_almost_equal(sources[0]['y'], pos[1], decimal=2)


@pytest.mark.flaky(max_runs=10, min_passes=1)
class Test_DAOFind_Detection():
    # DAOFind detection. Only round unsturaded stars

    def resort_sources(self, x, y, f):
        """Random sources are random. We must resort to compare."""
        # For SEP, resort using x order
        order = np.argsort(y)
        return x[order], y[order], f[order]

    @pytest.mark.parametrize('fwhm', np.arange(1, 9, 1))
    def test_daofind_constants_calc(self, fwhm):
        # Compare our constants with D.Jones daofind
        dao = DAOFind(fwhm)

        maxbox = 13  # Maximum size of convolution box in pixels
        assert_equal(dao._maxbox, maxbox)
        radius = np.max([0.637*fwhm, 2.001])
        assert_equal(dao._radius, radius)
        radsq = radius**2
        nhalf = np.min([int(radius), int((maxbox-1)/2.)])
        assert_equal(dao._nhalf, nhalf)
        nbox = 2*nhalf + 1  # number of pixels in side of convolution box
        assert_equal(dao._nbox, nbox)

        sigsq = (fwhm*gaussian_fwhm_to_sigma)**2
        assert_equal(dao._sigma2, sigsq)

        mask = np.zeros([nbox, nbox], dtype='int8')
        g = np.zeros([nbox, nbox])  # Gaussian convolution kernel
        row2 = (np.arange(nbox)-nhalf)**2
        for i in range(nhalf+1):
            temp = row2 + i**2
            g[nhalf-i, :] = temp
            g[nhalf+i, :] = temp
        g_row = np.where(g <= radsq)
        # MASK is complementary to SKIP in Stetson's Fortran
        mask[g_row[0], g_row[1]] = 1
        assert_equal(dao._conv_mask, mask)
        good = np.where(mask)  # Value of c are now equal to distance to center
        pixels = len(good[0])

        # Compute quantities for centroid computations that can be used for all
        # stars
        g = np.exp(-0.5*g/sigsq)
        assert_equal(dao._g, g)

        xwt = np.zeros([nbox, nbox])
        wt = nhalf - np.abs(np.arange(nbox)-nhalf) + 1
        assert_equal(dao._wt, wt)
        for i in range(nbox):
            xwt[i, :] = wt
        assert_equal(dao._xwt, xwt)
        ywt = np.transpose(xwt)
        assert_equal(dao._ywt, ywt)
        sgx = np.sum(g*xwt, 1)
        assert_equal(dao._sgx, sgx)
        p = np.sum(wt)
        assert_equal(dao._p, p)
        sgy = np.sum(g*ywt, 0)
        assert_equal(dao._sgy, sgy)
        sumgx = np.sum(wt*sgy)
        assert_equal(dao._sumgx, sumgx)
        sumgy = np.sum(wt*sgx)
        assert_equal(dao._sumgy, sumgy)
        sumgsqy = np.sum(wt*sgy*sgy)
        assert_equal(dao._sumgsqy, sumgsqy)
        sumgsqx = np.sum(wt*sgx*sgx)
        assert_equal(dao._sumgsqx, sumgsqx)
        vec = nhalf - np.arange(nbox)
        assert_equal(dao._vec, vec)
        dgdx = sgy*vec
        assert_equal(dao._dgdx, dgdx)
        dgdy = sgx*vec
        assert_equal(dao._dgdy, dgdy)
        sdgdxs = np.sum(wt*dgdx**2)
        assert_equal(dao._sdgdxs, sdgdxs)
        sdgdx = np.sum(wt*dgdx)
        assert_equal(dao._sdgdx, sdgdx)
        sdgdys = np.sum(wt*dgdy**2)
        assert_equal(dao._sdgdys, sdgdys)
        sdgdy = np.sum(wt*dgdy)
        assert_equal(dao._sdgdy, sdgdy)
        sgdgdx = np.sum(wt*sgy*dgdx)
        assert_equal(dao._sgdgdx, sgdgdx)
        sgdgdy = np.sum(wt*sgx*dgdy)
        assert_equal(dao._sgdgdy, sgdgdy)

        c = g*mask  # Convolution kernel now in c
        sumc = np.sum(c)
        sumcsq = np.sum(c**2) - sumc**2/pixels
        sumc = sumc/pixels
        c[good[0], good[1]] = (c[good[0], good[1]] - sumc)/sumcsq
        assert_equal(dao._kernel, c)
        c1 = np.exp(-.5*row2/sigsq)
        sumc1 = np.sum(c1)/nbox
        sumc1sq = np.sum(c1**2) - sumc1
        c1 = (c1-sumc1)/sumc1sq
        assert_equal(dao._c1, c1)

    def test_daofind_sharpness(self):
        # compare my implementation with D.Jones PythonPhot
        # without filtering, both have to output the same round/sharp and
        # the same coordinates for all stars, because use the same algorithm

        image_size = (525, 200)
        xpos = np.arange(10)*50 + 25
        ypos = np.ones_like(xpos)*30 + np.arange(len(xpos))*10
        sky = 800
        rdnoise = 50
        threshold = 50

        fwhm = 5
        sigma = np.array([0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.5, 3.0, 3.5])
        sigma *= fwhm*gaussian_fwhm_to_sigma
        flux = (np.ones_like(xpos)*sigma)*80000
        expect_sharp = [3.55, 0.76, 0.50, 0.45, 0.43,
                        0.41, 0.39, 0.39, 0.39, 0.39]

        im = gen_image(image_size, xpos, ypos, flux, sky, rdnoise,
                       model='gaussian', sigma=sigma)

        sources = daofind(im, threshold, sky, rdnoise, fwhm,
                          sharp_limit=None, round_limit=None)

        assert_equal(len(sources), len(xpos))
        assert_almost_equal(sources['x'], xpos, decimal=1)
        assert_almost_equal(sources['y'], ypos, decimal=1)
        assert_almost_equal(sources['sharp'], expect_sharp, decimal=1)

    def test_daofind_roundness(self):
        # compare my implementation with D.Jones PythonPhot
        # without filtering, both have to output the same round/sharp and
        # the same coordinates for all stars, because use the same algorithm

        image_size = (525, 200)
        xpos = np.arange(10)*50 + 25
        ypos = np.ones_like(xpos)*30 + np.arange(len(xpos))*10
        sky = 800
        rdnoise = 50
        threshold = 50

        e = (np.arange(len(xpos)))*0.1
        flux = np.sqrt(np.arange(len(xpos))+1)*80000
        sigma_x = np.sqrt(1/(1-e**2))*2
        sigma_y = np.ones_like(xpos)*2
        fwhm = 5
        theta = np.zeros_like(xpos)
        sky = 800
        rdnoise = 20  # very low noise
        expect_round = np.array([0.0, 0.02, 0.03, 0.08, 0.15, 0.25,
                                 0.39, 0.59, 0.88, 1.39])

        im = gen_image(image_size, xpos, ypos, flux, sky, rdnoise,
                       model='gaussian', sigma=(sigma_x, sigma_y),
                       theta=theta)

        sources = daofind(im, threshold, sky, rdnoise, fwhm,
                          sharp_limit=None, round_limit=None)

        assert_equal(len(sources), len(xpos))
        assert_almost_equal(sources['x'], xpos, decimal=0)
        assert_almost_equal(sources['y'], ypos, decimal=0)
        assert_almost_equal(sources['round'], -expect_round, decimal=1)

    def test_daofind_one_star(self):
        size = (128, 128)
        pos = (64, 64)
        sky = 70
        rdnoise = 20
        flux = 32000
        theta = 0
        fwhm = 3
        sigma = fwhm*gaussian_fwhm_to_sigma
        threshold = 10

        im = gen_image(size, [pos[0]], [pos[1]], [flux], sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = daofind(im, threshold, sky, rdnoise, fwhm)

        assert_equal(len(sources), 1)
        assert_almost_equal(sources['x'][0], pos[0], decimal=1)
        assert_almost_equal(sources['y'][0], pos[1], decimal=1)

    def test_daofind_strong_and_weak(self):
        size = (128, 128)
        posx = (45, 90)
        posy = (45, 90)
        sky = 800
        rdnoise = 20
        flux = (32000, 16000)
        fwhm = 3
        sigma = fwhm*gaussian_fwhm_to_sigma
        theta = 0
        im = gen_image(size, posx, posy, flux, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)
        threshold = 10

        sources = daofind(im, threshold, sky, rdnoise, fwhm)

        assert_equal(len(sources), 2)
        assert_almost_equal(sources['x'], posx, decimal=0)
        assert_almost_equal(sources['y'], posy, decimal=0)

    def test_daofind_four_stars_fixed_position(self):
        size = (128, 128)
        posx = (45, 90, 45, 90)
        posy = (45, 50, 90, 100)
        sky = 800
        rdnoise = 20
        flux = (15000, 3000, 5000, 35000)
        fwhm = 3
        sigma = fwhm*gaussian_fwhm_to_sigma
        theta = 0
        threshold = 10

        im = gen_image(size, posx, posy, flux, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = daofind(im, threshold, sky, rdnoise, fwhm)

        assert_equal(len(sources), 4)
        assert_almost_equal(sources['x'], posx, decimal=0)
        assert_almost_equal(sources['y'], posy, decimal=0)

    def test_daofind_multiple_stars(self):
        size = (512, 512)
        number = 15
        low = 2000
        high = 30000
        sky = 800
        rdnoise = 20
        fwhm = 5
        sigma = fwhm*gaussian_fwhm_to_sigma
        theta = 0
        threshold = 10
        x, y, f = gen_position_flux(size, number, low, high, rng_seed=456)

        im = gen_image(size, x, y, f, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = daofind(im, threshold, sky, rdnoise, fwhm)

        x, y, f = self.resort_sources(x, y, f)

        assert_equal(len(sources), number)
        assert_almost_equal(sources['x'], x, decimal=0)
        assert_almost_equal(sources['y'], y, decimal=0)

    def test_daofind_reject_roundness(self):
        size = (128, 128)
        pos_x = [20, 60, 100, 40, 80]
        pos_y = [20, 30, 40, 50, 60]
        sky = 70
        rdnoise = 20
        flux = [30000]*5
        theta = 0
        fwhm = 3
        sigma_x = np.array([1, 0.5, 1, 2.0, 0.1])*gaussian_fwhm_to_sigma*fwhm
        sigma_y = np.array([1, 1.0, 1, 0.5, 0.1])*gaussian_fwhm_to_sigma*fwhm
        threshold = 10
        # stars 0, 2 -> passed
        # star 4 -> rejected by sharpness
        # stars 1, 3 -> rejected by roundness

        im = gen_image(size, pos_x, pos_y, flux, sky, rdnoise,
                       model='gaussian', sigma=(sigma_x, sigma_y),
                       theta=theta)

        sources = daofind(im, threshold, sky, rdnoise, fwhm,
                          sharp_limit=(0.3, 0.6), round_limit=(-0.5, 0.5))

        assert_equal(len(sources), 2)
        assert_almost_equal(sources['x'], [20, 100], decimal=0)
        assert_almost_equal(sources['y'], [20, 40], decimal=0)

    def test_daofind_one_star_subpixel(self):
        size = (128, 128)
        pos = (54.32, 47.86)

        im = gen_image(size, [pos[0]], [pos[1]], [45000], 800, 0,
                       sigma=[5*gaussian_fwhm_to_sigma],
                       skip_poisson=True)
        sources = daofind(im, 5, 800, 10, 5)
        assert_equal(len(sources), 1)
        # no error, 2 decimals ok!
        assert_almost_equal(sources[0]['x'], pos[0], decimal=2)
        assert_almost_equal(sources[0]['y'], pos[1], decimal=2)


@pytest.mark.flaky(max_runs=10, min_passes=1)
class Test_StarFind():
    # Our combined iterative method

    def resort_sources(self, x, y, f):
        """Random sources are random. We must resort to compare."""
        # For SEP, resort using x order
        order = np.argsort(y)
        return x[order], y[order], f[order]

    def test_starfind_calc_fwhm(self):
        size = (512, 512)
        number = 15
        sky = 70
        rdnoise = 20
        low = 120000
        high = 320000
        sigma = 5
        theta = 0

        x, y, f = gen_position_flux(size, number, low, high, rng_seed=456)

        im = gen_image(size, x, y, f, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        fwhm = calc_fwhm(im, x, y, box_size=25, model='gaussian', min_fwhm=3.0)
        assert_almost_equal(fwhm, 2.35*sigma, decimal=0)

    pytest.mark.skip('Wrong new centers?')
    def test_starfind_recenter_sources(self):
        size = (256, 256)
        number = 10
        sky = 70
        rdnoise = 20
        low = 120000
        high = 320000
        sigma = 5
        theta = 0

        x, y, f = gen_position_flux(size, number, low, high, rng_seed=456)

        im = gen_image(size, x, y, f, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        nx, ny = recenter_sources(im,
                                  x+np.random.normal(loc=0, scale=1,
                                                     size=len(x)),
                                  y+np.random.normal(loc=0, scale=1,
                                                     size=len(x)),
                                  box_size=15, model='gaussian')

        assert_equal(len(nx), number)
        assert_equal(len(ny), number)
        # TODO: this seems unprecise. Investigate it.
        assert_almost_equal(nx, x, decimal=-1)
        assert_almost_equal(ny, y, decimal=-1)

    def test_starfind_one_star(self):
        size = (128, 128)
        x, y = (64, 64)
        f = 80000
        sky = 70
        rdnoise = 20
        sigma = 5
        theta = 0
        fwhm = 5  # dummy low value
        threshold = 10

        im = gen_image(size, [x], [y], [f], sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = starfind(im, threshold, sky, rdnoise, fwhm)

        assert_equal(len(sources), 1)
        assert_almost_equal(sources['x'], x, decimal=0)
        assert_almost_equal(sources['y'], y, decimal=0)
        assert_almost_equal(sources.meta['astropop fwhm'], sigma*2.355,
                            decimal=1)

    def test_starfind_strong_weak(self):
        size = (200, 100)
        posx = (50, 150)
        posy = (40, 60)
        sky = 800
        rdnoise = 20
        flux = (64000, 6000)
        theta = 0
        fwhm = 3
        sigma = fwhm*gaussian_fwhm_to_sigma
        threshold = 10
        im = gen_image(size, posx, posy, flux, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = starfind(im, threshold, sky, rdnoise, fwhm)

        assert_equal(len(sources), 2)
        assert_almost_equal(sources['x'], posx, decimal=1)
        assert_almost_equal(sources['y'], posy, decimal=1)
        assert_almost_equal(sources.meta['astropop fwhm'], sigma*2.355,
                            decimal=1)

    def test_starfind_blind_fwhm(self):
        size = (512, 512)
        number = 12
        low, high = (15000, 60000)
        sky = 800
        rdnoise = 20
        sigma = 2
        theta = 0
        threshold = 8

        x, y, f = gen_position_flux(size, number, low, high, rng_seed=456)
        x, y, f = self.resort_sources(x, y, f)

        im = gen_image(size, x, y, f, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = starfind(im, threshold, sky, rdnoise)

        assert_equal(len(sources), number)
        assert_almost_equal(sources['x'], x, decimal=1)
        assert_almost_equal(sources['y'], y, decimal=1)
        assert_almost_equal(sources.meta['astropop fwhm'], sigma*2.355,
                            decimal=2)

    def test_starfind_rejection(self):
        size = (128, 128)
        pos_x = [20, 60, 100, 40, 80, 50]
        pos_y = [20, 30, 40, 50, 60, 85]
        sky = 70
        rdnoise = 20
        flux = [30000]*6
        theta = 0
        fwhm = 3
        sig_base = gaussian_fwhm_to_sigma*fwhm
        sigma_x = np.array([1, 0.5, 1, 2.0, 0.1, 1])*sig_base
        sigma_y = np.array([1, 1.0, 1, 0.5, 0.1, 1])*sig_base
        threshold = 10
        # stars 0, 2 -> passed
        # star 4 -> rejected by sharpness
        # stars 1, 3 -> rejected by roundness

        im = gen_image(size, pos_x, pos_y, flux, sky, rdnoise,
                       model='gaussian', sigma=(sigma_x, sigma_y),
                       theta=theta)

        sources = starfind(im, threshold, sky, rdnoise,
                           sharp_limit=(0.3, 0.6), round_limit=(-0.5, 0.5))

        assert_equal(len(sources), 3)
        assert_almost_equal(sources['x'], [20, 100, 50], decimal=0)
        assert_almost_equal(sources['y'], [20, 40, 85], decimal=0)
        assert_almost_equal(sources.meta['astropop fwhm'], fwhm,
                            decimal=1)

    def test_starfind_one_star_subpixel(self):
        size = (128, 128)
        pos = (54.32, 47.86)

        im = gen_image(size, [pos[0]], [pos[1]], [45000], 800, 0,
                       sigma=[5*gaussian_fwhm_to_sigma],
                       skip_poisson=True)
        sources = starfind(im, 5, 800, 10)
        assert_equal(len(sources), 1)
        # no error, 2 decimals ok!
        assert_almost_equal(sources[0]['x'], pos[0], decimal=2)
        assert_almost_equal(sources[0]['y'], pos[1], decimal=2)
