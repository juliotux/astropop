# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import numpy as np
from scipy.integrate import dblquad, quad
from astropop.math.models import MoffatEquations, PSFMoffat1D, PSFMoffat2D, \
                                 PSFMoffatRadial
from astropop.math.models import GaussianEquations, PSFGaussian1D, \
                                 PSFGaussian2D, PSFGaussianRadial

from astropop.testing import *


class TestMoffatEquations:
    def test_moffat_2d_normalization(self):
        assert_almost_equal(MoffatEquations.normalize(2, 1.5),
                            0.5/(4*np.pi))

    def test_moffat_fwhm(self):
        assert_almost_equal(MoffatEquations.fwhm(2, 1.5),
                            2*2*np.sqrt(2**(1/1.5)-1))

    def test_moffat_radial_integral(self):
        v = quad(MoffatEquations.model_radial, a=0, b=np.inf,
                 args=(2, 1.5, 1, 0))[0]
        assert_almost_equal(v, 0.5)

    def test_moffat_1D_integral(self):
        v = quad(MoffatEquations.model_1d, a=-np.inf, b=np.inf,
                 args=(0, 2, 1.5, 1, 0))[0]
        assert_almost_equal(v, 1)

    def test_moffat_2D_integral(self):
        v = dblquad(MoffatEquations.model_2d, -np.inf, np.inf,
                    lambda x: -np.inf, lambda x: np.inf,
                    args=(0, 0, 2, 1.5, 1, 0))[0]
        assert_almost_equal(v, 1)


class TestGaussianEquations:
    def test_gaussian_2d_normalization(self):
        assert_almost_equal(GaussianEquations.normalize_2d(2, 2),
                            1/(2*np.pi*2*2))

    def test_gaussian_1d_normalization(self):
        assert_almost_equal(GaussianEquations.normalize_1d(2),
                            1/(np.sqrt(2*np.pi)*2))

    def test_gaussian_fwhm(self):
        assert_almost_equal(GaussianEquations.fwhm(2),
                            2*2*np.sqrt(2*np.log(2)))

    def test_gaussian_radial_integral(self):
        v = quad(GaussianEquations.model_radial, a=0, b=np.inf,
                 # f, sx, sky
                 args=(1, 2, 0))[0]
        assert_almost_equal(v, 0.5)

    def test_gaussian_1D_integral(self):
        v = quad(GaussianEquations.model_1d, a=-np.inf, b=np.inf,
                 # x0, f, sx, sky
                 args=(0, 1, 2, 0))[0]
        assert_almost_equal(v, 1)

    def test_gaussian_2D_integral(self):
        v = dblquad(GaussianEquations.model_2d, -np.inf, np.inf,
                    lambda x: -np.inf, lambda x: np.inf,
                    # x0, y0, f, sx, sy, theta, sky
                    args=(0, 0, 1, 2, 2, 0, 0))[0]
        assert_almost_equal(v, 1)


class TestMoffatModels:
    def test_model_radial_normalized(self):
        # moffat model need to be normalized to 1
        m = PSFMoffatRadial(flux=1, width=2, power=1.5, sky=0)
        i = quad(m, a=0, b=np.inf)[0]
        assert_almost_equal(i, 0.5)

    def test_model_1d_normalized(self):
        # moffat model need to be normalized to 1
        m = PSFMoffat1D(x_0=0, flux=1, width=2, power=1.5, sky=0)
        i = quad(m, a=-np.inf, b=np.inf)[0]
        assert_almost_equal(i, 1)

    def test_model_2d_normalized(self):
        # moffat model need to be normalized to 1
        m = PSFMoffat2D(x_0=0, y_0=0, flux=1, width=2, power=1.5, sky=0)
        i = dblquad(m, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]
        assert_almost_equal(i, 1)

    def test_model_radial_pnames(self):
        m = PSFMoffatRadial(flux=1, width=2, power=1.5, sky=0)
        # ensure names order
        assert_equal(m.param_names, ['flux', 'width', 'power', 'sky'])

    def test_model_1d_pnames(self):
        m = PSFMoffat1D(x_0=0, flux=1, width=2, power=1.5, sky=0)
        # ensure names order
        assert_equal(m.param_names, ['x_0', 'flux', 'width', 'power', 'sky'])

    def test_model_2d_pnames(self):
        m = PSFMoffat2D(x_0=0, y_0=0, flux=1, width=2, power=1.5, sky=0)
        # ensure names order
        assert_equal(m.param_names, ['x_0', 'y_0', 'flux',
                                     'width', 'power', 'sky'])

    def test_model_radial_fwhm(self):
        m = PSFMoffatRadial(flux=1, width=2, power=1.5, sky=0)
        assert_almost_equal(m.fwhm, 2*2*np.sqrt(2**(1/1.5)-1))

    def test_model_1d_fwhm(self):
        m = PSFMoffat1D(x_0=0, flux=1, width=2, power=1.5, sky=0)
        assert_almost_equal(m.fwhm, 2*2*np.sqrt(2**(1/1.5)-1))

    def test_model_2d_fwhm(self):
        m = PSFMoffat2D(x_0=0, y_0=0, flux=1, width=2, power=1.5, sky=0)
        assert_almost_equal(m.fwhm, 2*2*np.sqrt(2**(1/1.5)-1))


class TestGaussianModels:
    def test_model_radial_normalized(self):
        # gaussian model need to be normalized to 1
        m = PSFGaussianRadial(flux=1, sigma=2, sky=0)
        i = quad(m, a=0, b=np.inf)[0]
        assert_almost_equal(i, 0.5)

    def test_model_1d_normalized(self):
        # gaussian model need to be normalized to 1
        m = PSFGaussian1D(x_0=0, flux=1, sigma=2, sky=0)
        i = quad(m, a=-np.inf, b=np.inf)[0]
        assert_almost_equal(i, 1)

    def test_model_2d_normalized(self):
        # gaussian model need to be normalized to 1
        m = PSFGaussian2D(x_0=0, y_0=0, flux=1, sigma_x=2, sigma_y=2, sky=0)
        i = dblquad(m, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]
        assert_almost_equal(i, 1)

    def test_model_radial_pnames(self):
        m = PSFGaussianRadial(flux=1, sigma=2, sky=0)
        # ensure names order
        assert_equal(m.param_names, ['flux', 'sigma', 'sky'])

    def test_model_1d_pnames(self):
        m = PSFGaussian1D(x_0=0, flux=1, sigma=2, sky=0)
        # ensure names order
        assert_equal(m.param_names, ['x_0', 'flux', 'sigma', 'sky'])

    def test_model_2d_pnames(self):
        m = PSFGaussian2D(x_0=0, y_0=0, flux=1, sigma_x=2, sigma_y=2, sky=0)
        # ensure names order
        assert_equal(m.param_names, ['x_0', 'y_0', 'flux',
                                     'sigma_x', 'sigma_y',
                                     'theta', 'sky'])

    def test_model_radial_fwhm(self):
        m = PSFGaussianRadial(flux=1, sigma=2, sky=0)
        assert_almost_equal(m.fwhm, 2*(2*np.sqrt(2*np.log(2))))

    def test_model_1d_fwhm(self):
        m = PSFGaussian1D(x_0=0, flux=1, sigma=2, sky=0)
        assert_almost_equal(m.fwhm, 2*(2*np.sqrt(2*np.log(2))))

    def test_model_2d_fwhm(self):
        m = PSFGaussian2D(x_0=0, y_0=0, flux=1, sigma_x=2, sigma_y=2, sky=0)
        assert_almost_equal(m.fwhm, 2*(2*np.sqrt(2*np.log(2))))
        m = PSFGaussian2D(x_0=0, y_0=0, flux=1, sigma_x=2, sigma_y=1, sky=0)
        assert_almost_equal(m.fwhm, 1.5*(2*np.sqrt(2*np.log(2))))
