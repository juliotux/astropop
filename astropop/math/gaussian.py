# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gaussian models and utilities."""
import numpy as np
from astropy.modeling import Fittable1DModel, Fittable2DModel, Parameter
from astropy.stats import gaussian_sigma_to_fwhm


__all__ = ['gaussian_r', 'gaussian_1d', 'gaussian_2d', 'GaussianRadial',
           'PSFGaussian1D', 'PSFGaussian2D']


def gaussian_normalize(sigma, sigma2=None):
    """Normalize the gaussian distribution."""
    if sigma2 is None:
        sigma2 = sigma
    return 1.0/(np.sqrt(2*np.pi*sigma*sigma2))


def gaussian_fwhm(sigma, sigma2=None):
    """Get FWHM based on gaussian sigma."""
    if sigma2 is not None:
        return np.abs(gaussian_sigma_to_fwhm*np.hypot(sigma, sigma2))
    else:
        return np.abs(gaussian_sigma_to_fwhm*sigma)


def gaussian_r(r, sigma, flux, sky):
    a = flux*gaussian_normalize(sigma)
    return sky + a*np.exp(-0.5*(r/sigma)**2)


def gaussian_1d(x, x0, sigma, flux, sky):
    a = flux*gaussian_normalize(sigma)
    return sky + a*np.exp(-0.5*((x-x0)/sigma)**2)


def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, theta, flux, sky):
    cost2 = np.cos(np.radians(theta))**2
    sint2 = np.sin(np.radians(theta))**2
    sin2t = np.sin(2*np.radians(theta))
    xstd2 = sigma_x**2
    ystd2 = sigma_y**2
    a = (cost2/xstd2) + (sint2/ystd2)
    b = (sin2t/xstd2) - (sin2t/ystd2)
    c = (sint2/xstd2) + (cost2/ystd2)
    xi = x - x0
    yi = y - y0
    amp = flux*gaussian_normalize(sigma_x, sigma_y)
    return amp*np.exp(-0.5*((a*xi**2) + (b*xi*yi) + (c*yi**2)))


class PSFGaussian2D(Fittable2DModel):
    flux = Parameter(default=1, fixed=False)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    sigma_x = Parameter(default=1, fixed=False)
    sigma_y = Parameter(default=1, fixed=False)
    theta = Parameter(default=1, fixed=False)
    sky = Parameter(default=0, fixed=False)

    @staticmethod
    def evaluate(x, y, x_0, y_0, sigma_x, sigma_y, theta, flux, sky):
        return gaussian_2d(x, y, x_0, y_0, sigma_x, sigma_y, theta, flux, sky)

    @property
    def fwhm(self):
        return gaussian_fwhm(self.sigma_x, self.sigma_y)


class PSFGaussian1D(Fittable1DModel):
    flux = Parameter(default=1, fixed=False)
    x_0 = Parameter(default=0)
    sigma = Parameter(default=1, fixed=False)
    sky = Parameter(default=0, fixed=False)

    @staticmethod
    def evaluate(x, x_0, sigma, flux, sky):
        return gaussian_1d(x, x_0, sigma, flux, sky)

    @property
    def fwhm(self):
        return gaussian_fwhm(self.sigma)


class GaussianRadial(Fittable1DModel):
    flux = Parameter(default=1, fixed=False)
    sigma = Parameter(default=1, fixed=False)
    sky = Parameter(default=0, fixed=False)

    @staticmethod
    def evaluate(x, sigma, flux, sky):
        return gaussian_r(x, sigma, flux, sky)

    @property
    def fwhm(self):
        return gaussian_fwhm(self.sigma)
