from ._numba_helper import *

def _gaussian_r(r, sigma, flux, sky):
    return sky + (flux/(sqrt(2*pi*sigma**2)))*exp(-0.5*(r/sigma)**2)

def _gaussian_1d(x, x0, sigma, flux, sky):
    return sky + (flux/(sqrt(2*pi*sigma**2)))*exp(-0.5*((x-x0)/sigma)**2)

def _gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, theta, flux, sky):
    cost2 = cos(theta)**2
    sint2 = sin(theta)**2
    sin2t = sin(2*theta)
    sigx2 = 2*sigma_x**2
    sigy2 = 2*sigma_y**2
    a = (cost2/sigx2) + (sint2/sigy2)
    b = -(sin2t/(2*sigx2)) + (sin2t/(2*sigy2))
    c = (sint2/sigx2) + (cost2/sigy2)
    xi = x - x0
    yi = y - y0
    return sky + (flux/(sqrt(2*pi*sigma_x*sigma_y))) * exp(-(a*xi**2 + 2*b*xi*yi + c*yi**2))

if use_jit:
    gaussian_r = vectorize([i+str(tuple([i]*4)).replace('\'','') for i in ('float32', 'float64')],
                           target=numba_target)(_gaussian_r)
    gaussian_1d = vectorize([i+str(tuple([i]*5)).replace('\'','') for i in ('float32', 'float64')],
                            target=numba_target)(_gaussian_1d)
    gaussian_2d = vectorize([i+str(tuple([i]*9)).replace('\'','') for i in ('float32', 'float64')],
                            target=numba_target)(_gaussian_2d)
else:
    gaussian_r = _gaussian_r
    gaussian_1d = _gaussian_1d
    gaussian_2d = _gaussian_2d

__all__ = ['gaussian_r', 'gaussian_1d', 'gaussian_2d']
