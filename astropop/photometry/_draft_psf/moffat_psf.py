'''
Moffat optimizated kernels for psf fitting.
'''

from astropy.modeling import Fittable2DModel, Parameter
from math import log, pi
import numpy as np

from ._vectorize import vectorize, vectorize_target


@vectorize('float64(float64,float64,float64,float64,float64)',
           target=vectorize_target)
def rrgg(x, y, x_0, y_0, gamma):
    return ((x-x_0)**2 + (y-y_0)**2)/(gamma**2)


@vectorize('float64(float64,float64,float64,float64,float64,float64,float64)',
           target=vectorize_target)
def moffat2d(x, y, flux, x_0, y_0, gamma, alpha):
    rr_gg = ((x-x_0)**2 + (y-y_0)**2)/(gamma**2)
    return flux*((alpha-1)/(pi*gamma**2))*(1+rr_gg)**(-alpha)


@vectorize('float64(float64,float64,float64,float64)',
           target=vectorize_target)
def moffat2d_df(rr_gg, flux, gamma, alpha):
    return ((alpha-1)/(pi*gamma**2))*(1+rr_gg)**(-alpha)


@vectorize('float64(float64,float64,float64,float64,float64,float64)',
           target=vectorize_target)
def moffat2d_dx0(rr_gg, x, flux, x_0, gamma, alpha):
    return (-2*(alpha-1)*alpha*flux*(x-x_0)) * \
           ((rr_gg+1)**(-alpha-1))/(pi*(gamma**4))


@vectorize('float64(float64,float64,float64,float64)',
           target=vectorize_target)
def moffat2d_alpha(rr_gg, flux, gamma, alpha):
    return ((flux*(rr_gg+1)**(-alpha))/(pi*gamma**2)) * \
           (1 - (alpha - 1)*log(rr_gg+1))


@vectorize('float64(float64,float64,float64,float64)',
           target=vectorize_target)
def moffat2d_gamma(rr_gg, flux, gamma, alpha):
    return ((2*(alpha-1)*flux*(rr_gg+1)**(-alpha))/pi*gamma**3) * \
           ((alpha*rr_gg/(rr_gg+1)) - 1)


def moffat2d_deriv(x, y, flux, x_0, y_0, gamma, alpha):
    rr_gg = rrgg(x, y, x_0, y_0, gamma)
    return [moffat2d_df(rr_gg, flux, gamma, alpha),
            moffat2d_dx0(rr_gg, x, flux, x_0, gamma, alpha),
            moffat2d_dx0(rr_gg, y, flux, y_0, gamma, alpha),
            moffat2d_gamma(rr_gg, flux, gamma, alpha),
            moffat2d_alpha(rr_gg, flux, gamma, alpha)]


class Moffat2D_parallel(Fittable2DModel):
    '''
    Same results as the astropy.modeling.models.Moffat2D, but with
    numba jit and vectorized.
    '''
    flux = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    gamma = Parameter(default=1)
    alpha = Parameter(default=1)

    @staticmethod
    def evaluate(x, y, flux, x_0, y_0, gamma, alpha):
        return moffat2d(x, y, flux, x_0, y_0, gamma, alpha)

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, gamma, alpha):
        return moffat2d_deriv(x, y, flux, x_0, y_0, gamma, alpha)
