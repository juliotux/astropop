import numpy as np
from astropy.table import Table, hstack, vstack

from ..fits_utils import check_hdu
from ..photometry import (aperture_photometry, process_photometry,
                          solve_photometry_montecarlo,
                          solve_photometry_median,
                          solve_photometry_average,
                          starfind, background)
from .astrometry_scripts import solve_astrometry, identify_stars
from ..py_utils import check_iterable
from ..logger import logger


def solve_photometry(table, wcs=None, cat_mag=None,
                     identify_catalog=None, limit_angle='2 arcsec',
                     science_catalog=None, montecarlo_iters=100,
                     montecarlo_percentage=0.5, filter=None,
                     solve_photometry_type=None, flux_scale='linear',
                     cat_scale='mag'):
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
        id_table = identify_stars(table['x'], table['y'],
                                  wcs=wcs, filter=filter,
                                  identify_catalog=identify_catalog,
                                  science_catalog=science_catalog,
                                  limit_angle=limit_angle)
        cat_mag = id_table['cat_mag']

    mags = Table()

    if 'flux' in table.colnames:
        if identify_catalog is not None:
            cat_scale = identify_catalog.flux_unit
        solver_kwargs.update({'ref_scale': cat_scale,
                              'flux_scale': flux_scale})
        mags['mag'], mags['mag_err'] = solver(table['flux'],
                                              table['flux_error'],
                                              cat_mag, **solver_kwargs)

    return mags


def process_calib_photometry(image, wcs=None, return_wcs=False,
                             identify_catalog=None,
                             identify_limit_angle='2 arcsec',
                             science_catalog=None,
                             montecarlo_iters=100,
                             montecarlo_percentage=0.2, filter=None,
                             detect_snr=5, detect_fwhm=3,
                             solve_photometry_type=None, **kwargs):
    """Process photometry with magnitude calibration using catalogs."""
    image = check_hdu(image)

    result = Table()

    r = []
    if kwargs.get('photometry_type') in ['aperture', 'both']:
        if check_iterable(kwargs.get('r')):
            r = kwargs.get('r')
        else:
            r = [kwargs.get('r')]
    if kwargs.get('photometry_type') in ['psf', 'both']:
        r += ['psf']

    bkg, rms = background(image.data, 32, 3, global_bkg=False)
    sources = starfind(image.data, detect_snr, bkg, rms, fwhm=detect_fwhm)
    sources = aperture_photometry(image.data, sources['x'], sources['y'],
                                  r='auto', r_ann='auto')

    if wcs is None:
        logger.debug('Solving astrometry')
        wcs = solve_astrometry(sources, image.header,
                               image.data.shape,
                               ra_key=kwargs['ra_key'],
                               dec_key=kwargs['dec_key'],
                               plate_scale=kwargs['plate_scale'])
        logger.debug('WCS solved: {}'.format(wcs))

    photkwargs = {}
    for i in kwargs.keys():
        if i in ['detect_fwhm', 'detect_snr', 'box_size', 'r_ann',
                 'psf_model', 'psf_niters']:
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
        logger.debug('Identifying {} stars'.format(len(ph)))
        ids = identify_stars(ph['x'], ph['y'], wcs, filter=filter,
                             identify_catalog=identify_catalog,
                             limit_angle=identify_limit_angle,
                             science_catalog=science_catalog)
        ids = Table(ids)
        res = solve_photometry(ph, wcs, ids['cat_mag'],
                               montecarlo_iters=montecarlo_iters,
                               montecarlo_percentage=montecarlo_percentage,
                               solve_photometry_type=solve_photometry_type)

        t = hstack([ids, ph, res])

        result = vstack([result, t])

    if return_wcs:
        return result, wcs
    else:
        return result
