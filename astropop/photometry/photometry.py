# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.table import Table, hstack, vstack
from astropy.stats import gaussian_sigma_to_fwhm
import numpy as np

from ..fits_utils import check_hdu, imhdus
from ..logger import logger

try:
    from . import photutils_wrapper as phot
    _use_phot = True
except ModuleNotFoundError:
    _use_phot = False
    logger.warn('Photutils not found, ignoring it.')

try:
    from . import sep_wrapper as sep
    _use_sep = True
except ModuleNotFoundError:
    _use_sep = False
    logger.warn('SEP not found, ignoring it')


__all__ = ['aperture_photometry', 'process_photometry']


def aperture_photometry(data, detect_fwhm=None, detect_snr=None, x=None,
                        y=None, r=5, r_in=50, r_out=60, use_sep=True):
    """Perform aperture photometry in a image"""
    if isinstance(data, imhdus):
        data = data.data

    if use_sep and _use_sep:
        detect = sep.find_sources
        detect_kwargs = {}
        background = sep.calculate_background
        aperture = sep.aperture_photometry
    elif _use_phot:
        detect = phot.find_sources
        detect_kwargs = {'method': 'daofind', 'fwhm': detect_fwhm}
        background = phot.calculate_background
        aperture = phot.aperture_photometry
    else:
        raise ValueError('Sep and Photutils aren\'t installed. You must'
                         ' have at last one of them.')

    sky, rms = background(data)
    if x is not None and y is not None:
        s = np.array(list(zip(x, y)), dtype=[('x', 'f8'), ('y', 'f8')])
    else:
        s = detect(data, bkg=sky, rms=rms, snr=detect_snr, **detect_kwargs)
    ap = aperture(data, s['x'], s['y'], r, r_in, r_out, err=rms)
    res_ap = Table()
    res_ap['x'] = s['x']
    res_ap['y'] = s['y']
    res_ap['flux'] = ap['flux']
    res_ap['flux_error'] = ap['flux_error']

    return res_ap


def process_photometry(image, photometry_type, detect_fwhm=None,
                       detect_snr=None, box_size=None,
                       r=5, r_in=50, r_out=60, psf_model='gaussian',
                       psf_niters=1, x=None, y=None):
    """Process standart photometry in one image, without calibrations."""
    image = check_hdu(image)
    data = image.data

    if photometry_type == 'aperture':
        result = aperture_photometry(data, detect_fwhm=detect_fwhm,
                                     detect_snr=detect_snr, r=r,
                                     r_in=r_in, r_out=r_out,
                                     x=x, y=y)
        x = result['x']
        y = result['y']
    elif photometry_type == 'psf':
        if not _use_phot:
            raise ValueError('You must have Photutils installed for psf.')

        sigma = detect_fwhm/gaussian_sigma_to_fwhm
        ph = phot.psf_photometry(data, x, y, sigma_psf=sigma, snr=detect_snr,
                                 box_size=box_size, model=psf_model,
                                 niters=psf_niters)

        result = Table(ph)

    return result
