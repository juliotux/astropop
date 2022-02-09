# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from skimage import transform
from astropop.image.register import AsterismRegister, \
                                    CrossCorrelationRegister
from astropop.testing import assert_almost_equal

from .test_detection import gen_position_flux, gen_image


class Test_AsterismRegister:
    @pytest.mark.parametrize('shift', [(25, 32), (-12, 5), (23.42, 12.43)])
    def test_compute_transform_translation(self, shift):
        size = (1024, 1024)
        sky = 800
        n = 50
        rdnoise = 10
        x, y, f = gen_position_flux(size, n, 1e4, 4e6)
        sx, sy = shift

        # shift=(20x, 10y)
        im1 = gen_image(size, x, y, f, sky, rdnoise, sigma=2)
        im2 = gen_image(size, x+sx, y+sy, f, sky, rdnoise, sigma=2)

        ar = AsterismRegister()
        tform = ar._compute_transform(im1, im2)

        assert_almost_equal(tform.translation, shift, decimal=1)
        assert_almost_equal(tform.rotation, 0, decimal=3)
        assert_almost_equal(tform.scale, 1, decimal=4)

    def test_compute_transform_rotation(self):
        size = (1024, 1024)
        sky = 800
        n = 50
        rdnoise = 10
        x, y, f = gen_position_flux(size, n, 1e4, 4e6)

        im1 = gen_image(size, x, y, f, sky, rdnoise, sigma=2)
        im2 = transform.rotate(im1, -35.2,
                               preserve_range=True,
                               cval=im1.mean())

        ar = AsterismRegister()
        tform = ar._compute_transform(im1, im2)

        assert_almost_equal(tform.rotation, np.deg2rad(35.2), decimal=3)
        # the translation is needed due to the form skimage handles rotation
        assert_almost_equal(tform.translation, [388.37, -201.31], decimal=1)
        assert_almost_equal(tform.scale, 1, decimal=4)


class Test_CrossCorrelationRegister:
    @pytest.mark.parametrize('shift', [(25, 32), (-12, 5), (23.42, 12.43)])
    def test_compute_transform(self, shift):
        size = (1024, 1024)
        sky = 800
        n = 50
        rdnoise = 10
        x, y, f = gen_position_flux(size, n, 1e4, 4e6)
        sx, sy = shift

        im1 = gen_image(size, x, y, f, sky, rdnoise, sigma=2)
        im2 = gen_image(size, x+sx, y+sy, f, sky,
                        rdnoise, sigma=2)


        ccr = CrossCorrelationRegister(upsample_factor=3)
        tform = ccr._compute_transform(im1, im2)

        assert_almost_equal(tform.translation, shift, decimal=1)
        assert_almost_equal(tform.rotation, 0, decimal=3)
        assert_almost_equal(tform.scale, 1, decimal=4)
