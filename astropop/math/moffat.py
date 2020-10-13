# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Moffat models and utilities."""

import numpy as np
from astropy.modeling import Fittable1DModel, Fittable2DModel, Parameter
# from scipy.special import gamma as G


__all__ = ['moffat_r', 'moffat_1d', 'moffat_2d', 'PSFMoffat2D',
           'PSFMoffat1D', 'MoffatRadial']


def moffat_bounding_box(alpha, beta, threshold=10**-3):
    """Calculate the radius where the distribution reaches a given threshold.

    Threshold relative to maximum.
    """
    return int(np.sqrt((threshold**(-1/beta) - 1)*alpha))


def moffat_normalize(width, power):
    # From Yoonsoo Bach notebook
    # http://nbviewer.jupyter.org/github/ysbach/AO_2017/blob/master/04_Ground_Based_Concept.ipynb
    # return G(power)/(width*np.sqrt(np.pi)*G(power - 1/2))
    # Looking to the wikipedia, the PDF of Moffat distribution gives:
    return (power - 1)/(np.pi*width**2)


def moffat_fwhm(alpha, beta):
    return np.abs(2*alpha*np.sqrt(2**(1/beta) - 1))


def moffat_r(r, alpha, beta, flux, sky):
    sf = flux*moffat_normalize(alpha, beta)
    return sky + sf*(1+(r/alpha)**2)**(-beta)


def moffat_1d(x, x0, alpha, beta, flux, sky):
    r = x-x0
    return moffat_r(r, alpha, beta, flux, sky)


def moffat_2d(x, y, x0, y0, alpha, beta, flux, sky):
    # TODO: make this function assymetrical
    r = np.hypot(x-x0, y-y0)
    return moffat_r(r, alpha, beta, flux, sky)


class PSFMoffat2D(Fittable2DModel):
    flux = Parameter(default=1, fixed=False)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    alpha = Parameter(default=1, fixed=False)
    beta = Parameter(default=1.5, fixed=False)
    sky = Parameter(default=0, fixed=False)

    @staticmethod
    def evaluate(x, y, x_0, y_0, flux, alpha, beta, sky):
        return moffat_2d(x, y, x_0, y_0, alpha, beta, flux, sky)

    @property
    def fwhm(self):
        return moffat_fwhm(self.alpha, self.beta)


class PSFMoffat1D(Fittable1DModel):
    flux = Parameter(default=1, fixed=False)
    x_0 = Parameter(default=0)
    alpha = Parameter(default=1, fixed=False)
    beta = Parameter(default=1.5, fixed=False)
    sky = Parameter(default=0, fixed=False)

    @staticmethod
    def evaluate(x, x_0, flux, alpha, beta, sky):
        return moffat_1d(x, x_0, alpha, beta, flux, sky)

    @property
    def fwhm(self):
        return moffat_fwhm(self.alpha, self.beta)


class MoffatRadial(Fittable1DModel):
    flux = Parameter(default=1, fixed=False)
    alpha = Parameter(default=1, fixed=False)
    beta = Parameter(default=1.5, fixed=False)
    sky = Parameter(default=0, fixed=False)

    @staticmethod
    def evaluate(x, flux, alpha, beta, sky):
        return moffat_r(x, alpha, beta, flux, sky)

    @property
    def fwhm(self):
        return moffat_fwhm(self.alpha, self.beta)
