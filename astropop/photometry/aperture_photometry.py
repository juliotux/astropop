# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.table import Table, hstack, vstack
from astropy.stats import gaussian_sigma_to_fwhm
import numpy as np

from ..fits_utils import check_hdu, imhdus
from .solve_photometry import (solve_photometry_montecarlo,
                               solve_photometry_median,
                               solve_photometry_average)
from ..astrometry import solve_astrometry
from ..catalogs import identify_stars
from ..py_utils import check_iterable
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


def solve_photometry(table, wcs=None, cat_mag=None,
                     identify_catalog_file=None, identify_catalog_name=None,
                     identify_limit_angle='2 arcsec', science_catalog=None,
                     science_id_key=None, science_ra_key=None,
                     science_dec_key=None, montecarlo_iters=100,
                     montecarlo_percentage=0.5, filter=None,
                     solve_photometry_type=None):
    """Solve the absolute photometry of a field using a catalog."""
    if solve_photometry_type == 'montecarlo':
        solver = solve_photometry_montecarlo
        solver_kwargs = {'n_iter': montecarlo_iters,
                         'n_stars': montecarlo_percentage}
    elif solve_photometry_type == 'median':
        solver = solve_photometry_median
        solver_kwargs = {}
    elif solve_photometry_type == 'average':
        solver = solve_photometry_average
        solver_kwargs = {}
    else:
        raise ValueError('solve_photometry_type {} not'
                         ' supported'.format(solve_photometry_type))

    if cat_mag is None:
        id_table = identify_stars(table=table, wcs=wcs, filter=filter,
                                  identify_catalog_file=identify_catalog_file,
                                  identify_catalog_name=identify_catalog_name,
                                  science_catalog=science_catalog,
                                  science_id_key=science_id_key,
                                  science_ra_key=science_ra_key,
                                  science_dec_key=science_dec_key)
        cat_mag = id_table['cat_mag']

    mags = Table()

    if 'flux' in table.colnames:
        mags['mag'], mags['mag_err'] = solver(table['flux'],
                                              table['flux_error'],
                                              cat_mag, **solver_kwargs)

    return mags


def process_calib_photometry(image, identify_catalog_file=None,
                             identify_catalog_name=None,
                             identify_limit_angle='2 arcsec',
                             science_catalog=None, science_ra_key=None,
                             science_dec_key=None, science_id_key=None,
                             montecarlo_iters=100,
                             montecarlo_percentage=0.2, filter=None,
                             solve_photometry_type=None, **kwargs):
    """Process photometry with magnitude calibration using catalogs."""
    image = check_hdu(image)

    result = {'aperture': None, 'psf': None}

    r = []
    if kwargs.get('photometry_type') in ['aperture', 'both']:
        if check_iterable(kwargs.get('r')):
            r = kwargs.get('r')
        else:
            r = [kwargs.get('r')]
    if kwargs.get('photometry_type') in ['psf', 'both']:
        r += ['psf']

    sources = aperture_photometry(image.data, r=5,
                                  detect_snr=kwargs['detect_snr'],
                                  detect_fwhm=kwargs['detect_fwhm'])

    wcs = solve_astrometry(sources, image.header,
                           image.data.shape,
                           ra_key=kwargs['ra_key'],
                           dec_key=kwargs['dec_key'],
                           plate_scale=kwargs['plate_scale'])

    photkwargs = {}
    for i in kwargs.keys():
        if i in ['detect_fwhm', 'detect_snr', 'box_size',
                 'r_out', 'psf_model', 'psf_niters']:
            photkwargs[i] = kwargs.get(i)
    for ri in r:
        if ri == 'psf':
            phot_type = 'psf'
        else:
            phot_type = 'aperture'
        logger.info('Processing photometry for aperture {}'.format(ri))
        ph = process_photometry(image, r=ri, photometry_type=phot_type,
                                x=sources['x'], y=sources['y'],
                                **photkwargs)
        ids = identify_stars(ph, wcs, filter=filter,
                             identify_catalog_file=identify_catalog_file,
                             identify_catalog_name=identify_catalog_name,
                             identify_limit_angle=identify_limit_angle,
                             science_catalog=science_catalog,
                             science_id_key=science_id_key,
                             science_ra_key=science_ra_key,
                             science_dec_key=science_dec_key)
        res = solve_photometry(ph, wcs, ids['cat_mag'],
                               montecarlo_iters=montecarlo_iters,
                               montecarlo_percentage=montecarlo_percentage,
                               solve_photometry_type=solve_photometry_type)

        t = Table()
        t['star_index'] = np.arange(0, len(sources), 1)
        t['aperture'] = [ri if ri != 'psf' else np.nan]*len(sources)
        t = hstack([t, ids, ph, res])

        if result[phot_type] is None:
            result[phot_type] = t
        else:
            result[phot_type] = vstack([result[phot_type], t])

    return result
