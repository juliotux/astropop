# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from astropop.photometry.aperture import aperture_photometry, sky_annulus
from astropop.testing import *


@pytest.mark.parametrize('r', [2, 3, 4])
def test_simple_aperture(r):
    data = np.ones((10, 10))
    res = aperture_photometry(data, [5], [5], r=r, r_ann=None)
    assert_equal(res['x'], [5])
    assert_equal(res['y'], [5])
    assert_equal(res['aperture'], [r])
    assert_almost_equal(res['flux'], [np.pi*r**2])
    assert_not_in('sky', res.colnames)
    assert_almost_equal(res['flux_error'], [np.sqrt(np.pi*r**2)])
    assert_equal(res['flags'], [0])


data = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 2., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 2., 4., 2., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 2., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

@pytest.mark.parametrize(['algoritmo','sky_kind'],
                         [['mmm', 'simple'],['sigmaclip','simple'],
                          ['mmm', '2and4'],['sigmaclip','2and4']])
def test_sky_annulus(algoritmo, sky_kind):

    if sky_kind == 'simple':
        # Simple homogeneous test:
        data = np.ones((20, 20))*3
        data[10,10] = 10
        data[9,9:12] = 5
        data[10,9] = 5
        data[10,11] = 5
        data[11,9:12] = 5

        sky_target = np.ones((1,8))*3
        sky_error_target = np.zeros((1,8))

    elif sky_kind == '2and4':
        # Heterogeneous test:
        data = np.ones((20, 20))*2
        data[1,:20] = 4
        data[3,:20] = 4
        data[5,:20] = 4
        data[7,:20] = 4
        data[1,:20] = 4
        data[9,:20] = 4
        data[11,:20] = 4
        data[13,:20] = 4
        data[15,:20] = 4
        data[17,:20] = 4
        data[19,:20] = 4
        data[10,10] = 10
        data[9,9:12] = 5
        data[10,9] = 5
        data[10,11] = 5
        data[11,9:12] = 5

        sky_target = np.ones((1,8))*3
        sky_error_target = np.ones((1,8))

    x = [9,9,10,10,10,11,11,11]
    y = [10,11,9,10,11,9,10,11]
    r_ann = [3, 5]

    sky, sky_error = sky_annulus(data, x, y, r_ann, algoritmo) #estudar como testar logger e mask

    assert_equal([sky], sky_target)
    assert_equal([sky_error], sky_error_target)



# phototils - verificar casos testes para fotometrias
# phototils
# background
# tests

# devdolarpix do IRAF








