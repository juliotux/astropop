# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from astropop.polarimetry.dualbeam import match_pairs, estimate_dxdy, \
                                          _compute_theta, quarterwave_model, \
                                          halfwave_model, \
                                          _DualBeamPolarimetry, \
                                          SLSDualBeamPolarimetry, \
                                          StokesParameters
from astropop.math import QFloat
from astropy import units
from astropop.testing import *
from scipy.optimize import curve_fit
from functools import partial


def get_flux_oe(flux, psi, k, q, u, v=None, zero=0):
    """Get ordinary and extraordinary fluxes."""
    if v is None:
        zi = halfwave_model(psi, q, u, zero=zero)
    else:
        zi = quarterwave_model(psi, q, u, v, zero=zero)
    fo = flux*(1+zi)/2
    fe = flux*(1-zi)/2
    return fo, fe/k


class DummyPolarimeter(_DualBeamPolarimetry):
    def _half_fit(self, psi, zi):
        """Fit the Stokes params for halfwave retarder."""

    def _quarter_fit(self, psi, zi):
        """Fit the Stokes params for quarterwave retarder."""


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

        model = partial(quarterwave_model, zero=zero)

        fit, cov = curve_fit(model, psi, zi,
                             bounds=([-1, -1, -1], [1, 1, 1]),
                             method='trf')
        assert_almost_equal(fit, [q, u, v], decimal=3)

    def test_fit_free_zero(self):
        q = 0.0130
        u = -0.0021
        v = 0.03044
        zero = 60
        psi = np.arange(0, 360, 22.5)
        zi = quarterwave_model(psi, q=q, u=u, v=v, zero=zero)

        fit, cov = curve_fit(quarterwave_model, psi, zi,
                             bounds=([-1, -1, -1, 0], [1, 1, 1, 180]),
                             method='trf')
        assert_almost_equal(fit, [q, u, v, zero], decimal=3)


class Test_ModelHalf:
    def test_model_evaluate_plain(self):
        q = 0.0130
        u = -0.021
        zero = 60
        expect = np.array([0.01168653, 0.02175833, -0.01168653, -0.02175833,
                           0.01168653, 0.02175833, -0.01168653, -0.02175833,
                           0.01168653, 0.02175833, -0.01168653, -0.02175833,
                           0.01168653, 0.02175833, -0.01168653, -0.02175833])

        psi = np.arange(0, 360, 22.5)
        zi = halfwave_model(psi, q=q, u=u, zero=zero)
        assert_almost_equal(zi, expect)

    def test_model_evaluate_quantity(self):
        q = 0.0130
        u = -0.021
        expect = np.array([0.01168653, 0.02175833, -0.01168653, -0.02175833,
                           0.01168653, 0.02175833, -0.01168653, -0.02175833,
                           0.01168653, 0.02175833, -0.01168653, -0.02175833,
                           0.01168653, 0.02175833, -0.01168653, -0.02175833])

        psi = np.arange(0, 360, 22.5)*units.degree
        zero = 60*units.degree
        zi = halfwave_model(psi, q=q, u=u, zero=zero)
        assert_almost_equal(zi, expect)

        psi = np.arange(0, 2*np.pi, np.pi/8)*units.radian
        zero = 60*units.degree
        zi = halfwave_model(psi, q=q, u=u, zero=zero)
        assert_almost_equal(zi, expect)

    def test_fit(self):
        q = 0.0130
        u = -0.021
        zero = 60
        psi = np.arange(0, 360.5, 22.5)
        zi = halfwave_model(psi, q=q, u=u, zero=zero)

        model = partial(halfwave_model, zero=zero)

        fit, cov = curve_fit(model, psi, zi,
                             bounds=([-1, -1], [1, 1]),
                             method='trf')
        assert_almost_equal(fit, [q, u], decimal=3)


class Test_DummyPolarimetry:
    @pytest.mark.parametrize('kwargs', [{}, {'zero': 60},
                                        {'zero': 60, 'k': 1.2},
                                        {'zero': 60, 'k': 1.2, 'min_snr': 80},
                                        {'k': 1.0}, {'min_snr': 80},
                                        {'compute_k': True}])
    def test_initialize_ok(self, kwargs):
        pol = DummyPolarimeter('halfwave', **kwargs)
        assert_equal(pol.retarder, 'halfwave')
        if 'k' in kwargs:
            assert_equal(pol.k, kwargs['k'])
        else:
            assert_is_none(pol.k)
        if 'zero' in kwargs:
            assert_equal(pol.zero, kwargs['zero'])
        else:
            assert_is_none(pol.zero)
        if 'min_snr' in kwargs:
            assert_equal(pol.min_snr, kwargs['min_snr'])
        else:
            assert_is_none(pol.min_snr)
        if 'compute_k' in kwargs:
            assert_equal(pol.compute_k, kwargs['compute_k'])
        else:
            assert_false(pol.compute_k)

    def test_initialize_error_redundancy(self):
        with pytest.raises(ValueError, match='k and compute_k cannot be used '
                           'together.'):
            DummyPolarimeter('halfwave', k=1.2, compute_k=True)

    def test_initialize_error_retarder(self):
        with pytest.raises(ValueError, match="Retarder dummy unknown."):
            DummyPolarimeter('dummy')

    def test_initialize_zero_quantity(self):
        pol = DummyPolarimeter('halfwave', zero=60*units.degree)
        assert_equal(pol.zero, 60)

    def test_initialize_n_pos(self):
        pol = DummyPolarimeter('halfwave')
        assert_equal(pol._n_pos, 4)
        pol = DummyPolarimeter('quarterwave')
        assert_equal(pol._n_pos, 8)

    def test_estimate_normalize_half_ok(self):
        pol = DummyPolarimeter('halfwave')
        psi = np.arange(0, 360, 22.5)
        flux_o, flux_e = get_flux_oe(1e5, psi, k=1.2, q=0.0130,
                                     u=-0.021, zero=60)
        k = pol._estimate_normalize_half(psi, flux_o, flux_e)
        assert_almost_equal(k, 1.2)

    def test_estimate_normalize_half_error(self):
        pol = DummyPolarimeter('halfwave')
        psi = np.arange(0, 360, 22.5)

        flux_o, flux_e = get_flux_oe(1e5, psi, k=1.2, q=0.0130,
                                     u=-0.021, zero=60)
        flux_o[np.where(psi // 22.5 == 0)] = np.nan
        with pytest.raises(ValueError, match='Could not estimate the '
                           'normalization factor.'):
            pol._estimate_normalize_half(psi, flux_o, flux_e)

        flux_o, flux_e = get_flux_oe(1e5, psi, k=1.2, q=0.0130,
                                     u=-0.021, zero=60)
        flux_e[np.where(psi // 22.5 == 0)] = np.nan
        with pytest.raises(ValueError, match='Could not estimate the '
                           'normalization factor.'):
            pol._estimate_normalize_half(psi, flux_o, flux_e)

    def test_estimate_normalize_quarter_ok(self):
        q = 0.0130
        pol = DummyPolarimeter('halfwave')
        psi = np.arange(0, 360, 22.5)
        flux_o, flux_e = get_flux_oe(1e5, psi, k=1.2, q=0.0130,
                                     u=-0.021, zero=60)

        pol = DummyPolarimeter('quarterwave')
        k = pol._estimate_normalize_quarter(psi, flux_o, flux_e, q)
        assert_almost_equal(k,
                            np.sum(flux_o)/np.sum(flux_e)*(1-0.5*q)/(1+0.5*q))

    def test_check_positions(self):
        pol = DummyPolarimeter('halfwave')
        pol._check_positions(np.arange(0, 360, 22.5))

        psi = np.arange(0, 360, 22.5)
        psi[2] = 45.2
        with pytest.raises(ValueError, match="Retarder positions must be "
                           "multiple of 22.5 deg"):
            pol._check_positions(psi)

    def test_calc_zi(self):
        pol = DummyPolarimeter('halfwave')
        psi = np.arange(0, 360, 22.5)
        flux_o, flux_e = get_flux_oe(1e5, psi, k=1.2, q=0.0130,
                                     u=-0.021, zero=60)
        expect = halfwave_model(psi, q=0.0130, u=-0.021, zero=60)
        zi = pol._calc_zi(flux_o, flux_e, k=1.2)
        assert_almost_equal(zi, expect)


class Test_StokesParameters:
    def test_initialize_halfwave(self):
        q = QFloat(0.0130, 0.001)
        u = QFloat(-0.021, 0.001)

        # V is not needed
        pol = StokesParameters(retarder='halfwave', q=q, u=u)
        assert_equal(pol.q, q)
        assert_equal(pol.u, u)
        assert_is_none(pol.v)
        assert_equal(pol.retarder, 'halfwave')
        assert_equal(pol.k, 1.0)
        assert_equal(pol.zero, 0.0)

    def test_initialize_halfwave_kzero(self):
        q = QFloat(0.0130, 0.001)
        u = QFloat(-0.021, 0.001)
        k = QFloat(1.2, 0.001)
        zero = QFloat(60, 0.001)

        # V is not needed
        pol = StokesParameters(retarder='halfwave', q=q, u=u, k=k, zero=zero)
        assert_equal(pol.q, q)
        assert_equal(pol.u, u)
        assert_is_none(pol.v)
        assert_equal(pol.retarder, 'halfwave')
        assert_equal(pol.k, k)
        assert_equal(pol.zero, zero)

    def test_initialize_quarterwave(self):
        q = QFloat(0.0130, 0.001)
        u = QFloat(-0.021, 0.001)
        v = QFloat(0.021, 0.001)

        # V is needed
        pol = StokesParameters(retarder='quarterwave', q=q, u=u, v=v)
        assert_equal(pol.q, q)
        assert_equal(pol.u, u)
        assert_equal(pol.v, v)
        assert_equal(pol.retarder, 'quarterwave')
        assert_equal(pol.k, 1.0)
        assert_equal(pol.zero, 0.0)

    def test_initialize_quarterwave_kzero(self):
        q = QFloat(0.0130, 0.001)
        u = QFloat(-0.021, 0.001)
        v = QFloat(0.021, 0.001)
        k = QFloat(1.2, 0.001)
        zero = QFloat(60, 0.001)

        # V is needed
        pol = StokesParameters(retarder='quarterwave', q=q, u=u, v=v, k=k,
                               zero=zero)
        assert_equal(pol.q, q)
        assert_equal(pol.u, u)
        assert_equal(pol.v, v)
        assert_equal(pol.retarder, 'quarterwave')
        assert_equal(pol.k, k)
        assert_equal(pol.zero, zero)

    def test_stokes_properties(self):
        q = QFloat(0.0130, 0.001)
        u = QFloat(-0.021, 0.001)
        v = QFloat(0.021, 0.001)
        pol = StokesParameters(retarder='quarterwave', q=q, u=u, v=v)

        assert_almost_equal(pol.p.nominal, 0.0246981, decimal=3)
        assert_almost_equal(pol.theta.nominal, 150.8797, decimal=3)


class Test_SLSPolarimetry:
    def test_fit_half(self):
        q = 0.0130
        u = -0.021
        psi = np.arange(0, 360, 22.5)
        flux_o, flux_e = get_flux_oe(1e5, psi, k=1.0, q=q, u=u, zero=0)
        pol = SLSDualBeamPolarimetry(retarder='halfwave', k=1.0)
        p = pol.compute(psi, flux_o, flux_e,
                        f_ord_error=[50]*16, f_ext_error=[50]*16)
        assert_almost_equal(p.q.nominal, q)
        assert_almost_equal(p.u.nominal, u)
        assert_almost_equal(p.k, 1.0)
        assert_is_none(p.zero)

    def test_fit_half_no_errors(self):
        q = 0.0130
        u = -0.021
        psi = np.arange(0, 360, 22.5)
        flux_o, flux_e = get_flux_oe(1e5, psi, k=1.0, q=q, u=u, zero=0)
        pol = SLSDualBeamPolarimetry(retarder='halfwave', k=1.0)
        p = pol.compute(psi, flux_o, flux_e)
        assert_almost_equal(p.q.nominal, q)
        assert_almost_equal(p.u.nominal, u)
        assert_almost_equal(p.k, 1.0)
        assert_is_none(p.zero)

    def test_fit_half_estimate_k(self):
        q = 0.0130
        u = -0.021
        psi = np.arange(0, 360, 22.5)
        flux_o, flux_e = get_flux_oe(1e5, psi, k=1.2, q=q, u=u, zero=0)
        pol = SLSDualBeamPolarimetry(retarder='halfwave', compute_k=True)
        p = pol.compute(psi, flux_o, flux_e,
                        f_ord_error=[50]*16, f_ext_error=[50]*16)
        assert_almost_equal(p.q.nominal, q)
        assert_almost_equal(p.u.nominal, u)
        assert_almost_equal(p.k.nominal, 1.2)
        assert_is_none(p.zero)

    def test_fit_quarter(self):
        q = 0.0130
        u = -0.027
        v = 0.021
        zero = 60

        psi = np.arange(0, 360, 22.5)
        flux_o, flux_e = get_flux_oe(1e5, psi, k=1.0, q=q, u=u, v=v, zero=zero)
        pol = SLSDualBeamPolarimetry(retarder='quarterwave', zero=60,
                                     compute_k=True)
        p = pol.compute(psi, flux_o, flux_e,
                        f_ord_error=[50]*16, f_ext_error=[50]*16)

        assert_almost_equal(p.q.nominal, q)
        assert_almost_equal(p.u.nominal, u)
        assert_almost_equal(p.v.nominal, v)
        assert_almost_equal(p.k.nominal, 1.0, decimal=2)
        assert_almost_equal(p.zero, zero)

    def test_fit_quarter_no_errors(self):
        q = 0.0130
        u = -0.027
        v = 0.021
        zero = 60

        psi = np.arange(0, 360, 22.5)
        flux_o, flux_e = get_flux_oe(1e5, psi, k=1.0, q=q, u=u, v=v, zero=zero)
        pol = SLSDualBeamPolarimetry(retarder='quarterwave', zero=60,
                                     compute_k=True)
        p = pol.compute(psi, flux_o, flux_e)

        assert_almost_equal(p.q.nominal, q)
        assert_almost_equal(p.u.nominal, u)
        assert_almost_equal(p.v.nominal, v)
        assert_almost_equal(p.k.nominal, 1.0, decimal=2)
        assert_almost_equal(p.zero, zero)

    @pytest.mark.parametrize('k', [0.99, 1.01, 1.03, 0.97])
    def test_fit_quarter_estimate_k(self, k):
        q = 0.130
        u = -0.027
        v = 0.021
        zero = 60

        psi = np.arange(0, 360, 22.5)
        flux_o, flux_e = get_flux_oe(1e5, psi, k=k, q=q, u=u, v=v, zero=zero)
        pol = SLSDualBeamPolarimetry(retarder='quarterwave', zero=60,
                                     compute_k=True)
        p = pol.compute(psi, flux_o, flux_e)

        assert_almost_equal(p.q.nominal, q, decimal=3)
        assert_almost_equal(p.u.nominal, u, decimal=3)
        assert_almost_equal(p.v.nominal, v, decimal=3)
        assert_almost_equal(p.k.nominal, k, decimal=2)
        assert_almost_equal(p.zero, zero, decimal=3)

    def test_fit_quarter_estimate_zero(self):
        q = 0.130
        u = -0.027
        v = 0.021
        zero = 60

        psi = np.arange(0, 360, 22.5)
        flux_o, flux_e = get_flux_oe(1e5, psi, k=1.0, q=q, u=u, v=v, zero=zero)
        pol = SLSDualBeamPolarimetry(retarder='quarterwave', compute_k=True)
        p = pol.compute(psi, flux_o, flux_e)

        assert_almost_equal(p.q.nominal, q, decimal=3)
        assert_almost_equal(p.u.nominal, u, decimal=3)
        assert_almost_equal(p.v.nominal, v, decimal=3)
        assert_almost_equal(p.k.nominal, 1.0, decimal=2)
        assert_almost_equal(p.zero.nominal, zero, decimal=3)
