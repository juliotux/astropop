'''Standalone psf fitting functions. May be problematic.'''

# Fitting Functions
import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate
from astropy.nddata.utils import extract_array
from astropy.stats import sigma_clipped_stats

#import pyximport; pyximport.install()
from . import psf_kernels
from numba import autojit

@autojit
def compute_sky(z, sigma=2, mode='mean'):
    '''
    mode:
        mean: compute de mean of 33% lower values
        sigma_clip: compute the sigma_clipped stats and do the median of the
                    values between the lower value and n*sigma.
    '''
    if mode == 'mean':
        z = np.ravel(z)
        return np.mean(z[np.argsort(z)[:int(len(z)/3)]])
    elif mode == 'sigma_clip':
        mean, median, rms = sigma_clipped_stats(z)

        newz = np.ravel(z)
        return np.nanmedian(newz[newz < np.min(z) + sigma*rms])
    else:
        raise ValueError('Sky compute mode %s unrecognized.' % str(mode))

@autojit
def xy2r(x, y, data, xc, yc):
    r = np.sqrt((x-xc)**2 + (y-yc)**2)
    return np.ravel(r), np.ravel(data)

@autojit
def extract_data(data, indices, box_size, position):
    x, y = position
    dx = dy = float(box_size)/2

    x_min = max(int(x-dx), 0)
    x_max = int(x+dx)+1
    y_min = max(int(y-dy), 0)
    y_max = int(y+dy)+1

    d = data[y_min:y_max, x_min:x_max]
    xi = indices[1][y_min:y_max, x_min:x_max]
    yi = indices[0][y_min:y_max, x_min:x_max]
    return d, xi, yi

def generic_radial_fit(data, positions, flux, box_size,
                       dtype, function, nparams,
                       precalc_sky=True, sky_method='mean',
                       compute_errors=True):

    results = np.zeros(len(positions), dtype)
    if compute_errors:
        errors = np.zeros(len(positions), dtype)

    indices = np.indices(data.shape)
    for i in range(len(positions)):
        xp, yp = positions[i]
        d, xi, yi = extract_data(data, indices, box_size, (xp, yp))
        r, f = xy2r(xi, yi, d, xp, yp)

        if precalc_sky:
            sky = compute_sky(d, sky_method)
        else:
            sky = 0.0

        try:
            guess = tuple([flux[i]] + [1]*(nparams-2) + [sky])
            params, p_errors = curve_fit(function, r, f, p0=guess)
            p_errors = tuple([j for j in np.diag(p_errors)])
        except:
            nantuple = tuple([np.nan]*nparams)
            params, p_errors = nantuple, nantuple

        flux, flux_error = params[0], p_errors[0]
        r = tuple([xp, yp, flux, flux_error] + list(params) + list(p_errors))
        results[i] = np.array(r, dtype=dtype)

    return results

def generic_spatial_fit(data, positions, box_size,
                        dtype, function, nparams,
                        precalc_sky=True, sky_method='mean',
                        compute_errors=True):
    results = np.zeros(len(positions), dtype)
    if compute_errors:
        errors = np.zeros(len(positions), dtype)

    indices = np.indices(data.shape)
    for i in range(len(positions)):
        xp, yp = positions[i]
        d, xi, yi = extract_data(data, indices, box_size, (xp, yp))

        if precalc_sky:
            sky = compute_sky(d, sky_method)
        else:
            sky = 0.0

        try:
            guess = tuple([flux[i]] + [xp, yp] + [1]*(nparams-4) + [sky])
            params, p_errors = curve_fit(function, (xi, yi), np.ravel(d), p0=guess)
            p_errors = tuple([j for j in np.diag(p_errors)])
        except:
            nantuple = tuple([np.nan]*nparams)
            params, p_errors = nantuple, nantuple

        (xp, yp), (xp_err, yp_err) = params[0:2], p_errors[0:2]
        params, p_errors = params[2:] , p_errors[2:]

        flux, flux_error = params[0], p_errors[0]
        r = tuple([xp, yp, flux, flux_error] + list(params) + list(p_errors))
        results[i] = np.array(r, dtype=dtype)

    return results

#Gaussian functions#################################

def gaussian_radial(x, flux, sigma, sky):
    return psf_kernels.gaussian_radial(x, sigma, flux, sky)

def gaussian_spatial(xy, flux, x_0, y_0, sigma_x, sigma_y, theta, sky):
    x = xy[0]
    y = xy[1]
    return np.array(psf_kernels.gaussian_spatial(np.ravel(x), np.ravel(y), x_0, y_0, sigma_x, sigma_y, theta, flux, sky))

def fit_gaussian_radial(data, positions, flux, box_size,
                        precalc_sky=True, sky_method='mean',
                        compute_errors=True):
    dtype = np.dtype(list(zip(['x','y','flux','flux_error',
                          'flux_fit','sigma','sky',
                          'flux_fit_err', 'sigma_err', 'sky_err'],
                         ['f8']*10)))

    return generic_radial_fit(data, positions, flux, box_size,
                              dtype, gaussian_radial, 3,
                              precalc_sky=precalc_sky, sky_method=sky_method,
                              compute_errors=compute_errors)

def fit_gaussian_spatial(data, positions, flux, box_size,
                         precalc_sky=True, sky_method='mean',
                         compute_errors=True):
    dtype = np.dtype(list(zip(['x','y','flux','flux_error',
                          'flux_fit','sigmax','sigmay','theta','sky',
                          'flux_fit_err', 'sigmax_err','sigmay_err','theta_err', 'sky_err'],
                         ['f8']*14)))

    return generic_spatial_fit(data, positions, flux, box_size,
                               dtype, gaussian_spatial, 7,
                               precalc_sky=precalc_sky, sky_method=sky_method,
                               compute_errors=compute_errors)

#Moffat functions#################################

def moffat_radial(r, flux, gamma, alpha, sky):
    return psf_kernels.moffat_radial(r, gamma, alpha, flux, sky)

def moffat_spatial(xy, flux, x0, y0, gamma, alpha, sky):
    x = xy[0]
    y = xy[1]
    return np.array(psf_kernels.moffat_spatial(np.ravel(x), np.ravel(y), x0, y0, gamma, alpha, flux, sky)).ravel()

def fit_moffat_radial(data, positions, flux, box_size,
                       precalc_sky=True, sky_method='mean',
                       compute_errors=True):
    dtype = np.dtype(list(zip(['x','y','flux','flux_error',
                          'flux_fit','gamma','alpha','sky',
                          'flux_fit_err','gamma_err','alpha_err','sky_err'],
                         ['f8']*12)))

    return generic_radial_fit(data, positions, flux, box_size,
                              dtype, moffat_radial, 4,
                              precalc_sky=precalc_sky, sky_method=sky_method,
                              compute_errors=compute_errors)

def fit_moffat_spatial(data, positions, flux, box_size,
                       precalc_sky=True, sky_method='mean',
                       compute_errors=True):
    dtype = np.dtype(list(zip(['x','y','flux','flux_error',
                          'flux_fit','gamma','alpha','sky',
                          'flux_fit_err','gamma_err','alpha_err','sky_err'],
                         ['f8']*12)))

    return generic_spatial_fit(data, positions, flux, box_size,
                               dtype, moffat_spatial, 6,
                               precalc_sky=precalc_sky, sky_method=sky_method,
                               compute_errors=compute_errors)
