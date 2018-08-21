# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sep
import numpy as np
from astropy.table import Table

from ._utils import _sep_fix_byte_order
from .detection import calc_fwhm
from ..logger import logger
from ..fits_utils import imhdus


def aperture_photometry(data, x, y, r='auto', r_ann='auto', gain=None,
                        readnoise=None, mask=None, bkg_mask=None):
    """Perform aperture photometry using sep.

    Parameters:
    -----------
        - data : np.ndarray
            2D image data for photometry
        - x : array_like
            1D sequence of x positions of the sources
        - y : array_like
            1D sequence of y positions of the souces
        - r : float or 'auto' (optional)
            Aperture radius. If `auto`, the value will be estimated based in
            the median gaussian FWHM of the sources in the image
            (r=0.6731*GFWHM).
            Default: 'auto'
        - r_ann : array_like([float, float]), None or 'auto' (optional)
            Annulus radii (r_in, r_out) for local background extraction.
            If `auto`, the annulus will be set based on aperture as (4*r, 6*r).
            If None, no local background subtraction will be performed.
            Default: 'auto'
        - gain : float (optional)
            Gain to correctly calculate the error.
        - readnoise : float (optional)
            Readnoise of the image to correctly calculate the error.
        - mask : np.ndarray (optional)
            Mask badpixels and problematic ccd areas.
        - bkg_mask : np.ndarray (optional)
            Mask for sources during background extraction (avoid other sources
            contribution).
    """
    res_ap = Table()

    if isinstance(data, imhdus):
        data = data.data

    data = _sep_fix_byte_order(data)
    x = np.array(x)
    y = np.array(y)

    kwargs = {'gain': gain,
              'err': readnoise,
              'subpix': 0}

    if r == 'auto':
        logger.debug('Aperture r set as `auto`. Calculating from FWHM.')
        fwhm = calc_fwhm(data, x, y, box_size=25, model='gaussian')
        r = 0.6371*fwhm
        res_ap.meta['fwhm'] = fwhm
        logger.debug('FWHM:{} r:{}'.format(fwhm, r))

    res_ap.meta['r'] = r

    flux, flux_error, flags = sep.sum_circle(data, x, y, r, mask=mask,
                                             **kwargs)

    if r_ann == 'auto':
        logger.debug('Aperture r_ann set as `auto`. Calculating from r.')
        r_ann = (4*r, 6*r)
        logger.debug('r_ann:{}'.format(r_ann))

    if r_ann is not None:
        # We perform a custom background subtraction to make possible to mask
        # the sources
        ri, ro = r_ann
        res_ap.meta['r_in'] = ri
        res_ap.meta['r_out'] = ro
        if bkg_mask is not None:
            bkg_mask |= mask
        else:
            bkg_mask = mask
        flua, erroa, flaga = sep.sum_circann(data, x, y, ri, ro, mask=bkg_mask,
                                            **kwargs)

        # SEP do not expose aperture area, so we calculate
        nkwargs = kwargs.copy()
        nkwargs['gain'] = 1.0 # ensure gain not change the area
        ones = _sep_fix_byte_order(np.ones(data.shape, dtype='<f8'))
        area, _, _ = sep.sum_circle(ones, x, y, r, mask=mask, **kwargs)
        area1, _, _ = sep.sum_circann(ones, x, y, ri, ro, mask=bkg_mask,
                                      **kwargs)

        # recompute flux, flux_error and flags
        flux -= flua*(area/area1)
        bkgerr = erroa*(area/area1)
        flux_error = np.hypot(flux_error, bkgerr)

    res_ap['x'] = x
    res_ap['y'] = y
    res_ap['aperture'] = [r]*len(x)
    res_ap['flux'] = flux
    res_ap['flux_error'] = flux_error
    res_ap['flags'] = flags

    return res_ap
