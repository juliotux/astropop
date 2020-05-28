# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
import numpy.testing as npt
import pytest_check as check
from astropop.photometry.aperture import aperture_photometry


@pytest.mark.parametrize('r', [2, 3, 4])
def test_simple_aperture(r):
    data = np.ones((10, 10))
    res = aperture_photometry(data, [5], [5], r=r, r_ann=None)
    npt.assert_array_equal(res['x'], [5])
    npt.assert_array_equal(res['y'], [5])
    npt.assert_array_equal(res['aperture'], [r])
    npt.assert_almost_equal(res['flux'], [np.pi*r**2])
    check.is_not_in('sky', res.colnames)
    npt.assert_array_almost_equal(res['flux_error'], [np.sqrt(np.pi*r**2)])
    npt.assert_array_equal(res['flags'], [0])


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