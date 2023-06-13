# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import numpy as np
from astropop.photometry.aperture import aperture_photometry
from astropop.testing import *


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
