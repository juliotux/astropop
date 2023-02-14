# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sep
import numpy as np
from astropy.table import Table

from ._utils import _sep_fix_byte_order
from .detection import calc_fwhm
from ..logger import logger
from ..fits_utils import imhdus


class PhotometryFlags:
    """Flags for aperture photometry.

    For subclassing, please set values higher than 1 << 4
    """
    # 1: At least one pixel in the aperture was removed
    REMOVED_PIXEL_IN_APERTURE = 1 << 0
    # 2: At least one pixel in the aperture was interpolated
    INTERPOLATED_PIXEL_IN_APERTURE = 1 << 1
    # 4: At least one pixel in the aperture was out of bounds of the image
    OUT_OF_BOUNDS = 1 << 2


def aperture_photometry(data, x, y, r='auto', r_ann='auto', gain=1.0,
                        readnoise=None, mask=None, sky_algorithm='mmm'):
    """Perform aperture photometry using sep. Output units (ADU or e-) will
    be the same as input.

    Parameters
    ----------
    data : `~nnumpy.ndarray`
        2D image data for photometry
    x, y : array_like
        Positions of the sources
    r : float or 'auto' (optional)
        Aperture radius. If 'auto', the value will be estimated based in
        the median gaussian FWHM of the sources in the image
        (r=0.6731*GFWHM).
        Default: 'auto'
    r_ann : array_like([float, float]), `None` or 'auto' (optional)
        Annulus radii (r_in, r_out) for local background extraction.
        If 'auto', the annulus will be set based on aperture as (4*r, 6*r).
        If None, no local background subtraction will be performed.
        Default: 'auto'
    gain : float (optional)
        Gain to correctly calculate the error.
        Default: 1.0
    readnoise : float (optional)
        Readnoise of the image to correctly calculate the error.
        Default: `None`
    mask : `~numpy.ndarray` (optional)
        Mask badpixels and problematic ccd areas.
        Default: `None`
    sky_algorithm : 'mmm' or 'sigmaclip'
        Algorith to calculate the background value. 'mmm' (mean, median,
        mode) should be better for populated fields, while 'sigmaclip'
        (clipped median) should be better for sparse fields.
        Default: 'mmm'

    Returns
    -------
    res_ap : `~astropy.table.Table`
        Table containing all aperture photometry informations.
        - ``x``, ``y``: position of the sources
        - ``aperture``: aperture radius
        - ``flux``: flux of the sources with bkg subtracted
        - ``flux_error``: flux error of the sources with bkg subtracted
        - ``sky``: sky background value by pixel
        - ``sky_error``: sky background error by pixel
        - ``flag``: flag for the sources
    """
    res_ap = Table()

    if isinstance(data, imhdus):
        data = data.data

    data = _sep_fix_byte_order(data)
    x = np.array(x)
    y = np.array(y)

    if gain is None:
        gain = 1.0
    kwargs = {'gain': gain,
              'err': readnoise,
              'subpix': 0}

    res_ap['x'] = x
    res_ap['y'] = y

    if r == 'auto':
        logger.debug('Aperture r set as `auto`. Calculating from FWHM.')
        fwhm = calc_fwhm(data, x, y, box_size=25, model='gaussian')
        r = 0.6371*fwhm
        res_ap.meta['fwhm'] = fwhm
        res_ap.meta['r_auto'] = True
        logger.debug(f'FWHM:{fwhm} r:{r}')

    # res_ap['fwhm'] = fwhm
    res_ap['aperture'] = [r]*len(x)

    if r_ann == 'auto':
        logger.debug('Aperture r_ann set as `auto`. Calculating from r.')
        r_in = int(round(4*r, 0))
        r_out = int(max(r_in+10, round(6*r, 0)))  # Ensure a dannulus geq 10
        r_ann = (r_in, r_out)
        logger.debug(f'r_ann:{r_ann}')

    sky = None

    flux, flux_error, flags = sep.sum_circle(data, x, y, r, mask=mask,
                                             **kwargs)
    if r_ann is not None:
        ri, ro = r_ann
        res_ap.meta['r_in'] = ri
        res_ap.meta['r_out'] = ro

        sky, error = sky_annulus(data, x, y, r_ann, algorithm=sky_algorithm,
                                 mask=mask)

        # SEP do not expose aperture area, so we calculate
        nkwargs = kwargs.copy()
        nkwargs['gain'] = 1.0  # ensure gain not change the area
        ones = _sep_fix_byte_order(np.ones(data.shape, dtype='<f8'))
        area, _, _ = sep.sum_circle(ones, x, y, r, mask=mask, **nkwargs)

        # TODO: check these calculations
        # recompute flux, flux_error and flags
        flux -= sky*area
        bkgerr = np.sqrt((error + np.absolute(sky)/gain)*area)
        flux_error = np.hypot(flux_error, bkgerr)

    res_ap['flux'] = flux
    res_ap['flux_error'] = flux_error
    res_ap['flags'] = flags

    if sky is not None:
        res_ap['sky'] = sky
        res_ap['sky_error'] = error

    return res_ap
