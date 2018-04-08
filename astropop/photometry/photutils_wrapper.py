# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from astropy.stats import sigma_clipped_stats
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import Table

from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry as photutils_aperture
from photutils import DAOStarFinder, IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOPhotPSFPhotometry

# from ..logger import logger


def find_sources(data, fwhm, threshold=None, bkg=0.0, rms=None, snr=None,
                 method='daofind', **kwargs):
    '''method : ('daofind', 'iraf')

    sky will be subtracted from the data

    kwargs are DAOStarFinder or IRAFStarFinder, like sharphi, sharplo, roundhi,
    roundlo
    '''

    if threshold is None:
        if rms is None or snr is None:
            raise ValueError('You must give a threshold or bkg, snr and rms.')
        else:
            threshold = snr*rms

    if method == 'daofind':
        find = DAOStarFinder(fwhm=fwhm, threshold=threshold, sky=bkg,
                             exclude_border=True, **kwargs)
    elif method == 'iraf':
        find = IRAFStarFinder(fwhm=fwhm, threshold=threshold, sky=bkg,
                              exclude_border=True, **kwargs)
    else:
        raise ValueError('Method {} unrecognized.'.format(method))
    i = find(data)

    s_dt = np.dtype([('x', 'f8'), ('y', 'f8'), ('flux', 'f8'),
                     ('sharpness', 'f8'), ('roundness', 'f8'), ('sky', 'f8'),
                     ('peak', 'f8')])
    return np.array(list(zip(i['xcentroid'], i['ycentroid'], i['flux'],
                             i['sharpness'], i['roundness1'], i['sky'],
                             i['peak'])), s_dt)


def calculate_background(data, sigma=3, iters=1):
    mean, median, std = sigma_clipped_stats(data, sigma=sigma, iters=iters)
    return 2.5*median - 1.5*mean, std


def aperture_photometry(data, x, y, r, r_in, r_out, err=None, **kwargs):
    ap_c = CircularAperture(zip(x, y), r=r)
    ap_a = CircularAnnulus(zip(x, y), r_in=r_in, r_out=r_out)
    ap = photutils_aperture(data, ap_c, error=err, **kwargs)
    ann = photutils_aperture(data, ap_a, error=err, **kwargs)

    flux = ap['aperture_sum'] - (ap_c.area()/ap_a.area())*ann['aperture_sum']

    p_dt = np.dtype([('x', 'f8'), ('y', 'f8'), ('flux', 'f8'),
                     ('flux_error', 'f8')])
    return np.array(list(zip(x, y, flux, ap['aperture_sum_err'])), p_dt)


def psf_photometry(data, x=None, y=None, sigma_psf=1.0, snr=10, box_size=20,
                   model='gaussian', niters=1):
    '''Perform the PSF photometry using photutils DAOphot algorith.
    '''

    bkg, rms = calculate_background(data, 3, 1)

    photargs = {'crit_separation': sigma_psf*5,
                'threshold': snr*rms,
                'fwhm': sigma_psf*gaussian_sigma_to_fwhm,
                'aperture_radius': sigma_psf*gaussian_sigma_to_fwhm,
                'fitter': LevMarLSQFitter(),
                'niters': niters,
                'sharplo': 0.0,
                'sharphi': 2.0,
                'roundlo': -2.0,
                'roundhi': 2.0,
                'fitshape': (box_size, box_size)}

    if model == 'gaussian':
        photargs['psf_model'] = IntegratedGaussianPRF(sigma=sigma_psf)
        photargs['psf_model'].sigma.fixed = False
    # elif model == 'moffat':
    #     photargs['psf_model'] = Moffat2D(alpha=0.5, gamma=1.5)
    #     photargs['psf_model'].alpha.fixed = False
    else:
        raise ValueError('Model not supported.')

    phot = DAOPhotPSFPhotometry(**photargs)

    if x is None or y is None:
        res = phot(data)
    else:
        res = phot(data, positions=Table(names=['x_0', 'y_0'], data=[x, y]))

    r_dt = np.dtype([('x', 'f8'), ('y', 'f8'), ('flux', 'f8'),
                     ('flux_error', 'f8')])
    return np.array(list(zip(res['x_fit'], res['y_fit'], res['flux_fit'],
                             [np.nan]*len(res))), dtype=r_dt)
