# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np

from astropop.photometry import (background, segfind, daofind, starfind,
                                 median_fwhm)
from astropop.math.models import MoffatEquations, GaussianEquations
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


def gen_position_flux(size, number, low, high, rng_seed=123, fwhm=5):
    """Generate x, y, and flux lists for stars."""
    for i in range(number):
        with NumpyRNGContext(rng_seed):
            x = np.random.randint(fwhm, size[0]-fwhm, number)
        with NumpyRNGContext(rng_seed+i):
            y = np.random.randint(fwhm, size[1]-fwhm, number)
    # lets sample the flux in the range. Avoid tests flakinness
    step = float(high-low)/number
    flux = np.arange(number)*step + low
    return np.array(x), np.array(y), np.sort(flux)[::-1]


def gen_stars_moffat(size, x, y, flux, fwhm):
    """Generate stars image to add to background."""
    power = 1.5
    width = 0.5*fwhm/np.sqrt(2**(1/power)-1)

    im = np.zeros(size[::-1])
    grid_y, grid_x = np.indices(size[::-1])
    for xi, yi, fi in zip(x, y, flux):
        imi, gxi, gyi = trim_array(np.zeros_like(im), box_size=5*fwhm,
                                   position=(xi, yi),
                                   indices=(grid_y, grid_x))
        imi += MoffatEquations.model_2d(gxi, gyi, xi, yi, flux=fi,
                                        width=width, power=power, sky=0)
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

    bc = np.broadcast_arrays(x, y, flux, sigma_x, sigma_y, theta)

    for xi, yi, fi, sx, sy, ti in zip(*bc):
        imi, gxi, gyi = trim_array(np.zeros_like(im), box_size=5*max(sx, sy),
                                   position=(xi, yi),
                                   indices=(grid_y, grid_x))
        imi += GaussianEquations.model_2d(gxi, gyi, xi, yi, flux=fi,
                                          sigma_x=sx, sigma_y=sy, theta=ti,
                                          sky=0)
        im[gyi.min():gyi.max()+1, gxi.min():gxi.max()+1] += imi

    return im


def gen_image(size, x, y, flux, sky, rdnoise, model='gaussian', **kwargs):
    """Generate a full image of stars with noise."""
    im = np.ones(size)*sky
    if rdnoise > 0:
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


@pytest.mark.flaky(reruns=5, reruns_delay=0.1)
class Test_Background():
    def test_background_unkown_methods(self):
        # unkown methods should fail
        size = (2048, 2048)
        level = 800
        rdnoise = 20
        image_test = gen_bkg(size, level, rdnoise)

        box_size = 64
        filter_size = 3
        for i in ['unkown', None, 1, 'average']:
            with pytest.raises(ValueError,
                               match='Unknown background method'):
                background(image_test, box_size, filter_size,
                           bkg_method='unkown')
            with pytest.raises(ValueError,
                               match='Unknown rms method'):
                background(image_test, box_size, filter_size,
                           rms_method=i)

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

    @pytest.mark.parametrize('global_bkg', [True, False])
    @pytest.mark.parametrize('method', ['mean', 'median', 'mode'])
    def test_background_no_changes_inplace(self, method, global_bkg):
        # check if background changes the default image inplace.
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

        image_test_o = image_test.copy()
        background(image_test, 64, 3, bkg_method=method, global_bkg=global_bkg)
        assert_equal(image_test_o, image_test)

    def test_background_touple_sclip(self):
        size = (1024, 1024)
        level = 800
        rdnoise = 20
        image_test = gen_bkg(size, level, rdnoise)

        box_size = 64
        filter_size = 3
        global_bkg, global_rms = background(image_test, box_size, filter_size,
                                            mask=None, global_bkg=True)
        bkg, rms = background(image_test, box_size, filter_size,
                              mask=None, global_bkg=False,
                              sigma_clip=(3, 3))

        assert_equal(type(global_bkg), float)
        assert_equal(type(global_rms), float)
        assert_almost_equal(global_bkg, level, decimal=0)
        assert_almost_equal(global_rms, rdnoise, decimal=0)

        assert_equal(bkg.shape, size)
        assert_equal(rms.shape, size)
        assert_almost_equal(bkg, np.ones(size)*level, decimal=0)
        assert_almost_equal(rms, np.ones(size)*rdnoise, decimal=0)


@pytest.mark.flaky(reruns=5, reruns_delay=0.1)
class Test_Segmentation_Detection():
    # segmentation detection. Must detect all shapes of sources

    def test_segfind_one_star(self):
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

        sources = segfind(im, threshold, sky, rdnoise)

        assert_equal(len(sources), 1)
        assert_almost_equal(sources['x'][0], 64, decimal=0)
        assert_almost_equal(sources['y'][0], 64, decimal=0)

    def test_segfind_negative_sky(self):
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

        sources = segfind(im, threshold, sky, rdnoise)

        assert_equal(len(sources), 1)
        assert_almost_equal(sources['x'][0], 64, decimal=0)
        assert_almost_equal(sources['y'][0], 64, decimal=0)

    def test_segfind_strong_and_weak(self):
        size = (128, 128)
        posx = (60, 90)
        posy = (20, 90)
        sky = 800
        rdnoise = 20
        flux = (32000, 3000)
        sigma = 1.5
        theta = 0
        im = gen_image(size, posx, posy, flux, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = segfind(im, 5, sky, rdnoise)

        assert_almost_equal(sources['x'], posx, decimal=0)
        assert_almost_equal(sources['y'], posy, decimal=0)

    def test_segfind_four_stars_fixed_position(self):
        size = (1024, 1024)
        posx = (10, 120, 500, 1000)
        posy = (20, 200, 600, 800)
        sky = 800
        rdnoise = 20
        flux = (35000, 15000, 10000, 5000)
        sigma = 1.5
        theta = 0
        im = gen_image(size, posx, posy, flux, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = segfind(im, 10, sky, rdnoise)

        assert_almost_equal(sources['x'], posx, decimal=0)
        assert_almost_equal(sources['y'], posy, decimal=0)

    def test_segfind_multiple_stars(self):
        size = (1024, 1024)
        number = 15
        low = 5000
        high = 30000
        sky = 800
        rdnoise = 20
        sigma = 1.5
        theta = 0

        x, y, f = gen_position_flux(size, number, low, high, rng_seed=456)
        im = gen_image(size, x, y, f, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = segfind(im, 8, sky, rdnoise)

        assert_almost_equal(sources['x'], x, decimal=0)
        assert_almost_equal(sources['y'], y, decimal=0)

    def test_segfind_multiple_stars_with_kernel_convolution(self):
        size = (1024, 1024)
        number = 15
        low = 5000
        high = 30000
        sky = 800
        rdnoise = 20
        sigma = 1.5
        theta = 0
        fwhm = 2

        x, y, f = gen_position_flux(size, number, low, high, rng_seed=456)
        im = gen_image(size, x, y, f, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = segfind(im, 8, sky, rdnoise, fwhm=fwhm)

        assert_almost_equal(sources['x'], x, decimal=0)
        assert_almost_equal(sources['y'], y, decimal=0)

    def test_segfind_one_star_subpixel(self):
        size = (128, 128)
        pos = (54.32, 47.86)

        im = gen_image(size, [pos[0]], [pos[1]], [45000], 800, 0,
                       sigma=[3], skip_poisson=True)
        sources = segfind(im, 5, 800, 10)
        assert_equal(len(sources), 1)
        # no error, 2 decimals ok!
        assert_almost_equal(sources[0]['x'], pos[0], decimal=2)
        assert_almost_equal(sources[0]['y'], pos[1], decimal=2)

    def test_segfind_no_changes_inplace(self):
        # check if segfind changes the default image inplace.
        size = (128, 128)
        pos = (54.32, 47.86)

        im = gen_image(size, [pos[0]], [pos[1]], [45000], 800, 0,
                       sigma=[5*gaussian_fwhm_to_sigma],
                       skip_poisson=True)
        im_o = im.copy()
        sources = segfind(im, 5, 800, 10)
        assert_equal(im_o, im)


@pytest.mark.flaky(reruns=5, rerun_delay=0.1)
class Test_DAOFind_Detection():
    # DAOFind detection. Only round unsturaded stars

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
        sigma = np.array([0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5])
        sigma *= fwhm*gaussian_fwhm_to_sigma
        sigma = sigma[::-1]
        flux = np.ones_like(xpos)*80000/GaussianEquations.normalize_2d(sigma,
                                                                       sigma)
        expect_sharp = [3.5, 0.8, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4][::-1]

        im = gen_image(image_size, xpos, ypos, flux, sky, rdnoise,
                       model='gaussian', sigma=sigma)

        sources = daofind(im, threshold, sky, rdnoise, fwhm,
                          sharp_limit=None, round_limit=None)

        assert_almost_equal(sources['x'], xpos, decimal=1)
        assert_almost_equal(sources['y'], ypos, decimal=1)
        assert_almost_equal(sources['sharpness'], expect_sharp, decimal=1)

    def test_daofind_roundness(self):
        image_size = (525, 200)
        xpos = np.arange(10)*50 + 25
        ypos = np.ones_like(xpos)*30 + np.arange(len(xpos))*10
        sky = 800
        rdnoise = 50
        threshold = 50

        e = (np.arange(len(xpos)))*0.1
        flux = (np.sqrt(np.arange(len(xpos))+1)*80000)[::-1]
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
        assert_almost_equal(sources['roundness'], expect_round, decimal=1)

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
        flux = (35000, 15000, 10000, 5000)
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
        low = 5000
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

        assert_almost_equal(sources['x'], x, decimal=0)
        assert_almost_equal(sources['y'], y, decimal=0)

    def test_daofind_reject_roundness(self):
        size = (128, 128)
        pos_x = [20, 60, 100, 40, 80]
        pos_y = [20, 30, 40, 50, 60]
        sky = 70
        rdnoise = 20
        flux = np.array([3e5, 3e5, 2e5, 3e5, 3e5])  # ensure order
        theta = 0
        fwhm = 3
        sigma_x = np.array([1, 0.5, 0.9, 2.0, 0.1])*gaussian_fwhm_to_sigma*fwhm
        sigma_y = np.array([1, 1.0, 0.9, 0.5, 0.1])*gaussian_fwhm_to_sigma*fwhm
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

    def test_daofind_no_changes_inplace(self):
        # check if daofind changes the default image inplace.
        size = (128, 128)
        pos = (54.32, 47.86)

        im = gen_image(size, [pos[0]], [pos[1]], [45000], 800, 0,
                       sigma=[5*gaussian_fwhm_to_sigma],
                       skip_poisson=True)
        im_o = im.copy()
        sources = daofind(im, 5, 800, 10, 5)
        assert_equal(im_o, im)


@pytest.mark.flaky(reruns=5, rerun_delay=0.1)
class Test_StarFind():
    # Our combined iterative method

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

        fwhm = median_fwhm(im, x, y, box_size=25, model='gaussian',
                           min_fwhm=3.0)
        assert_almost_equal(fwhm, 2.35*sigma, decimal=0)

    def test_starfind_calc_fwhm_moffat(self):
        size = (512, 512)
        number = 15
        sky = 70
        rdnoise = 20
        low = 120000
        high = 320000
        fwhm_true = 5

        x, y, f = gen_position_flux(size, number, low, high, rng_seed=456)

        im = gen_image(size, x, y, f, sky, rdnoise,
                       model='moffat', fwhm=fwhm_true)

        fwhm = median_fwhm(im, x, y, box_size=25, model='moffat',
                           min_fwhm=3.0)
        assert_almost_equal(fwhm, fwhm_true, decimal=0)

    def test_starfind_one_star(self):
        size = (128, 128)
        x, y = (64, 64)
        f = 80000
        sky = 70
        rdnoise = 20
        sigma = 5
        theta = 0
        threshold = 10

        im = gen_image(size, [x], [y], [f], sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = starfind(im, threshold, sky, rdnoise, fwhm=3.0)

        assert_equal(len(sources), 1)
        assert_almost_equal(sources['x'], x, decimal=0)
        assert_almost_equal(sources['y'], y, decimal=0)
        assert_almost_equal(sources.meta['astropop fwhm'], sigma*2.355,
                            decimal=0)

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
                            decimal=0)

    def test_starfind_blind_fwhm(self):
        size = (512, 512)
        number = 12
        low, high = (15000, 60000)
        sky = 800
        rdnoise = 20
        sigma = 2
        theta = 0
        threshold = 10

        x, y, f = gen_position_flux(size, number, low, high, rng_seed=456)

        im = gen_image(size, x, y, f, sky, rdnoise,
                       model='gaussian', sigma=sigma, theta=theta)

        sources = starfind(im, threshold, sky, rdnoise)

        assert_equal(len(sources), number)
        assert_almost_equal(sources['x'], x, decimal=1)
        assert_almost_equal(sources['y'], y, decimal=1)
        assert_almost_equal(sources.meta['astropop fwhm'], sigma*2.355,
                            decimal=0)

    def test_starfind_rejection(self):
        size = (128, 128)
        pos_x = np.array([20, 60, 100, 40, 80, 50])
        pos_y = np.array([20, 30, 40, 50, 60, 85])
        sky = 70
        rdnoise = 5
        flux = np.arange(6)*500+30000
        theta = 0
        fwhm = 3
        sig_base = gaussian_fwhm_to_sigma*fwhm
        sigma_x = np.array([1, 0.5, 1.2, 2.0, 0.1, 0.8])*sig_base
        sigma_y = np.array([1, 1.0, 1.2, 0.5, 0.1, 0.9])*sig_base
        threshold = 10
        # stars 0, 2, 5 -> passed
        # star 4 -> rejected by sharpness
        # stars 1, 3 -> rejected by roundness
        order = [5, 2, 0]  # flux encreases in order, peak not

        im = gen_image(size, pos_x, pos_y, flux, sky, rdnoise,
                       model='gaussian', sigma=(sigma_x, sigma_y),
                       theta=theta)

        sources = starfind(im, threshold, sky, rdnoise,
                           sharp_limit=(0.3, 0.6), round_limit=(-0.5, 0.5))

        assert_equal(len(sources), 3)
        assert_almost_equal(sources['x'], pos_x[order], decimal=0)
        assert_almost_equal(sources['y'], pos_y[order], decimal=0)
        assert_almost_equal(sources.meta['astropop fwhm'], fwhm,
                            decimal=0)

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

    def test_starfind_no_changes_inplace(self):
        # check if daofind changes the default image inplace.
        size = (128, 128)
        pos = (54.32, 47.86)

        im = gen_image(size, [pos[0]], [pos[1]], [45000], 800, 0,
                       sigma=[5*gaussian_fwhm_to_sigma],
                       skip_poisson=True)
        im_o = im.copy()
        sources = starfind(im, 5, 800, 10)
        assert_equal(im_o, im)
