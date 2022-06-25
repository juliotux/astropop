# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from astropop.polarimetry.dualbeam import match_pairs, estimate_dxdy, \
                                          compute_theta
from astropop.testing import *


def test_compute_theta():
    assert_almost_equal(compute_theta(0, 1), 45)
    assert_almost_equal(compute_theta(1, 0), 0)
    assert_almost_equal(compute_theta(0, -1), 135)
    assert_almost_equal(compute_theta(-1, 0), 90)
    assert_almost_equal(compute_theta(0.5, 0.5), 22.5)
    assert_almost_equal(compute_theta(0.5, -0.5), 157.5)
    assert_almost_equal(compute_theta(-0.5, 0.5), 67.5)
    assert_almost_equal(compute_theta(-0.5, -0.5), 112.5)


class Test_PairMatching:
    def test_estimate_dxdy(self):
        dx, dy = 30.5, -25.2
        x1 = np.random.uniform(low=0, high=1024, size=500)
        y1 = np.random.uniform(low=0, high=1024, size=500)
        x2 = x1 + dx
        y2 = y1 + dy

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))

        dx_e, dy_e = estimate_dxdy(x, y, steps=[50, 5, 1], bins=100)

        # 1 decimal is enough
        assert_almost_equal(dx, dx_e, decimal=1)
        assert_almost_equal(dy, dy_e, decimal=1)

    def test_match_pairs(self):
        dx, dy = 30.5, -25.2

        x1 = np.array([46.7, 68.3, 131.9, 83.5, 34.7, 170.2, 96.0, 115.9,
                       155.6, 25.8])
        y1 = np.array([186.5, 29.9, 94.9, 105.2, 43.4, 108.8, 177.6, 18.5, 5.4,
                       119.6])
        x2 = x1 + dx
        y2 = y1 + dy

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))

        index = match_pairs(x, y, dx, dy, 0.5)

        assert_equal(len(index), 10)
        assert_equal(index['o'], np.arange(10, 20))
        assert_equal(index['e'], np.arange(0, 10))

    def test_match_pairs_negative(self):
        dx, dy = 30.5, -25.2

        x1 = np.array([46.7, 68.3, 131.9, 83.5, 34.7, 170.2, 96.0, 115.9,
                       155.6, 25.8])
        y1 = np.array([186.5, 29.9, 94.9, 105.2, 43.4, 108.8, 177.6, 18.5, 5.4,
                       119.6])
        x2 = x1 - dx
        y2 = y1 - dy

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))

        index = match_pairs(x, y, dx, dy, 0.5)

        assert_equal(len(index), 10)
        assert_equal(index['o'], np.arange(0, 10))
        assert_equal(index['e'], np.arange(10, 20))
