# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from astropop.polarimetry.dualbeam import match_pairs, estimate_dxdy, \
                                          _compute_theta, quarterwave_model
from astropy import units
from astropy.modeling import fitting
from astropop.testing import *


def test_compute_theta():
    assert_equal(_compute_theta(0, 1), 45*units.degree)
    assert_equal(_compute_theta(1, 0), 0*units.degree)
    assert_equal(_compute_theta(0, -1), 135*units.degree)
    assert_equal(_compute_theta(-1, 0), 90*units.degree)
    assert_equal(_compute_theta(0.5, 0.5), 22.5*units.degree)
    assert_equal(_compute_theta(0.5, -0.5), 157.5*units.degree)
    assert_equal(_compute_theta(-0.5, 0.5), 67.5*units.degree)
    assert_equal(_compute_theta(-0.5, -0.5), 112.5*units.degree)


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


class Test_ModelQuarter:
    def test_model_evaluate_plain(self):
        q = 0.0130
        u = -0.0021
        v = 0.03044
        zero = 60
        # values simulated that match IRAF pccdpack
        expect = np.array([-0.02220249, 0.00477571, 0.02406067, 0.02974862,
                           0.03052114, 0.02053262, -0.00637933, -0.02905695,
                           -0.02220249, 0.00477571, 0.02406067, 0.02974862,
                           0.03052114, 0.02053262, -0.00637933, -0.02905695])

        psi = np.arange(0, 360, 22.5)
        zi = quarterwave_model(psi, q=q, u=u, v=v, zero=zero)
        assert_almost_equal(zi, expect)

    def test_model_evaluate_quantity(self):
        q = 0.0130
        u = -0.0021
        v = 0.03044
        # values simulated that match IRAF pccdpack
        expect = np.array([-0.02220249, 0.00477571, 0.02406067, 0.02974862,
                           0.03052114, 0.02053262, -0.00637933, -0.02905695,
                           -0.02220249, 0.00477571, 0.02406067, 0.02974862,
                           0.03052114, 0.02053262, -0.00637933, -0.02905695])

        psi = np.arange(0, 360, 22.5)*units.degree
        zero = 60*units.degree
        zi = quarterwave_model(psi, q=q, u=u, v=v, zero=zero)
        assert_almost_equal(zi, expect)

        psi = np.arange(0, 2*np.pi, np.pi/8)*units.radian
        zero = 60*units.degree
        zi = quarterwave_model(psi, q=q, u=u, v=v, zero=zero)
        assert_almost_equal(zi, expect)

    def test_fit(self):
        q = 0.0130
        u = -0.0021
        v = 0.03044
        zero = 60
        psi = np.arange(0, 360.5, 22.5)
        zi = quarterwave_model(psi, q=q, u=u, v=v, zero=zero)

        # fitter = fitting.LevMarLSQFitter()
        # m_fit = fitter(QuarterWaveModel(zero=60), pos, zi)
        # assert_almost_equal(m_fit.q, q)
        # assert_almost_equal(m_fit.u, u)
        # assert_almost_equal(m_fit.v, v)
        # assert_almost_equal(m_fit.zero, zero)

    def test_fit_free_zero(self):
        q = 0.0130
        u = -0.0021
        v = 0.03044
        zero = 60
        psi = np.arange(0, 360, 22.5)
        zi = quarterwave_model(psi, q=q, u=u, v=v, zero=zero)

        # fitter = fitting.TRFLSQFitter()
        # m_fit = fitter(QuarterWaveModel(fixed={'zero': False}), pos, zi)
        # assert_almost_equal(m_fit.q, q)
        # assert_almost_equal(m_fit.u, u)
        # assert_almost_equal(m_fit.v, v)
        # assert_almost_equal(m_fit.zero, zero)
