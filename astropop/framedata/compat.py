# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Handle compatibility between FrameData and other data formats."""

import warnings
import itertools
from os import PathLike
import copy as cp
import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata.ccddata import CCDData
from astropy.nddata import StdDevUncertainty, VarianceUncertainty, \
                           InverseVariance
from astropy import units as u


from ..logger import logger
from ..fits_utils import imhdus
from ..py_utils import broadcast


__all__ = ['imhdus', 'EmptyDataError']


_PCs = set(['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'])
_CDs = set(['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'])
_KEEP = set(['JD-OBS', 'MJD-OBS', 'DATE-OBS'])
_HDU_UNCERT = 'UNCERT'
_HDU_MASK = 'MASK'
_UNIT_KEY = 'BUNIT'


def _remove_sip_keys(header, wcs):
    """Remove the SIP keys. (Too complex fixing)."""
    kwd = '{}_{}_{}'
    pol = ['A', 'B', 'AP', 'BP']
    for poly in pol:
        order = wcs.sip.__getattribute__(f'{poly.lower()}_order')
        for i in [f'{poly}_ORDER', f'{poly}_DMAX']:
            if i in header.keys():
                header.remove(i)
        for i, j in itertools.product(range(order), repeat=2):
            key = kwd.format(poly, i, j)
            if key in header.keys():
                header.remove(key)
    return header


def extract_header_wcs(header):
    """Get a header (or dict) and extract a WCS based on the keys.

    Parameters
    ----------
    header: `~astropy.fits.Header` or dict_like
        Header to extract the WCS.
    logger: `~logging.Logger` (optional)
        Logger instance to log warnings.

    Returns
    -------
    header: `~astropy.fits.Header`
        Header cleaned from WCS keys.
    wcs: `~astropy.wcs.WCS` os `None`
        World Coordinate Sistem extracted from the header.
    """
    header = header.copy()

    # First, check if there is a WCS. If not, return header and None WCS
    try:
        with warnings.catch_warnings():
            # silent WCS anoying warnings
            warnings.filterwarnings("ignore", category=FITSFixedWarning)
            wcs = WCS(header, relax=True, fix=True)
            if not wcs.wcs.ctype[0]:
                wcs = None
    except Exception as e:
        logger.warning('An error raised when extracting WCS: %s', e)
        wcs = None

    if wcs is not None:
        # Delete wcs keys, except time ones
        for k in wcs.to_header(relax=True).keys():
            if k not in _KEEP and k in header.keys():
                header.remove(k)

        # Astropy uses PC. So remove CD if any
        # The reverse case should not be allowed by astropy
        if (_PCs & set(wcs.to_header(relax=True))) and \
           (_CDs & set(header.keys())):
            for k in _CDs:
                if k in header.keys():
                    header.remove(k)

        # Check and remove remaining SIP coefficients
        if wcs.sip is not None:
            header = _remove_sip_keys(header, wcs)

    return (header, wcs)


def _merge_and_clean_header(meta, header, wcs):
    """Merge meta and header and clean the WCS and spurious keys."""
    for i in (meta, header, wcs):
        if not isinstance(i, (dict, fits.Header, WCS)) and i is not None:
            raise TypeError(f'{i} is not a compatible format.')

    if header is not None:
        header = fits.Header(header)
        header.strip()
    else:
        header = fits.Header()

    if meta is not None:
        meta = fits.Header(meta)
        meta.strip()
    else:
        meta = fits.Header()

    meta.update(header)

    # strip blank cards
    meta.remove('', ignore_missing=True)

    # extract history and comments from meta
    if 'history' in meta:
        history = meta['history']
        history = list(broadcast(history).iters[0])
        del meta['history']
    else:
        history = []

    if 'comment' in meta:
        comment = meta['comment']
        comment = list(broadcast(comment).iters[0])
        del meta['comment']
    else:
        comment = []

    # extract wcs from header
    meta, wcs_ = extract_header_wcs(meta)
    wcs = wcs_ if wcs is None else wcs
    return meta, wcs, history, comment


class EmptyDataError(ValueError):
    """Operation not handled by empty MemMapArray containers."""


def _extract_fits(obj, hdu=0, unit=None, hdu_uncertainty=_HDU_UNCERT,
                  hdu_mask=_HDU_MASK, unit_key=_UNIT_KEY):
    """Extract data and meta from FITS files and HDUs."""
    if isinstance(obj, (str, bytes, PathLike)):
        hdul = fits.open(obj)
    elif isinstance(obj, imhdus):
        hdul = fits.HDUList([obj])
    elif isinstance(obj, fits.HDUList):
        hdul = obj
    else:
        raise TypeError(f'Object {obj} not recognized.')

    # from now, assumes obj is HDUList
    # Check if the data HDU contain data
    if hdul[hdu].data is None and isinstance(hdul[hdu], imhdus):
        if hdu == 0:
            hdu_data = None
        else:
            raise ValueError('Informed data HDU do not contain any data.')
    else:
        hdu_data = hdu

    # scan hdus for the first with image data if the first hdu don't
    for ind in range(len(hdul)):
        hdu_i = hdul[ind]
        if not isinstance(hdu_i, imhdus) or hdu_i.data is None:
            continue
        hdu_data = ind
        if hdu_data != hdu:
            logger.info('First hdu with image data: %i', ind)
        break

    hdu_data = hdul[hdu_data]
    res = {}
    res['data'] = hdu_data.data
    res['meta'] = hdu_data.header

    unit_h = res['meta'].get(unit_key, None)
    if unit_h is None or unit_h == '':
        res['unit'] = unit
    else:
        try:
            unit_h = u.format.Fits.parse(unit_h)
        except ValueError:
            unit_h = u.Unit(unit_h)
        if unit_h != unit and unit is not None:
            raise ValueError('unit and unit_key got incompatible results.')
        res['unit'] = unit_h

    if hdu_uncertainty in hdul:
        hdu_unct = hdul[hdu_uncertainty]
        logger.debug('Uncertainty hdu found. Assuming standard uncertainty.')
        res['uncertainty'] = hdu_unct.data

    if hdu_mask in hdul:
        hdu_mask = hdul[hdu_mask]
        res['mask'] = hdu_mask.data

    if isinstance(obj, (str, bytes, PathLike)):
        hdul.close()

    return res


def _extract_ccddata(ccd):
    """Extract data and meta from CCDData objects."""
    res = {}
    for i in ('data', 'unit', 'meta', 'mask'):
        res[i] = cp.copy(getattr(ccd, i))

    uunit = None
    uncert = ccd.uncertainty
    if uncert is None:
        uncert = None
    elif isinstance(uncert, StdDevUncertainty):
        uunit = uncert.unit
        uncert = uncert.array
    elif isinstance(uncert, VarianceUncertainty):
        uunit = uncert.unit**0.5
        uncert = np.sqrt(uncert.array)
    elif isinstance(uncert, InverseVariance):
        uunit = uncert.unit**(-0.5)
        uncert = np.sqrt(1/uncert.array)
    else:
        raise TypeError(f'Unrecognized uncertainty type {uncert.__class__}.')

    if uunit != res['unit'] and uunit is not None:
        raise ValueError(f'Uncertainty with unit {uunit} different from the '
                         f'data {res["unit"]}')
    res['uncertainty'] = uncert

    return res


def _to_ccddata(frame):
    """Translate a FrameData to a CCDData."""
    data = np.array(frame.data)
    unit = frame.unit
    meta = dict(frame.header)
    wcs = frame.wcs
    uncertainty = frame._unct
    if uncertainty.empty:
        uncertainty = None
    else:
        uncertainty = StdDevUncertainty(uncertainty, unit=unit)
    mask = np.array(frame.mask)

    return CCDData(data, unit=unit, meta=meta, wcs=wcs,
                   uncertainty=uncertainty, mask=mask)


def _to_hdu(frame, hdu_uncertainty=_HDU_UNCERT, hdu_mask=_HDU_MASK,
            unit_key=_UNIT_KEY, wcs_relax=True, **kwargs):
    """Translate a FrameData to an HDUList."""
    data = frame.data.copy()

    # Clean header
    header = fits.Header(frame.header)
    no_fits_std = kwargs.pop('no_fits_standard_units', False)

    if frame.wcs is not None:
        header.extend(frame.wcs.to_header(relax=wcs_relax),
                      update=True)

    if no_fits_std:
        unit_string = frame.unit.to_string()
    else:
        unit_string = u.format.Fits.to_string(frame.unit)

    header[unit_key] = unit_string

    for i in frame.history:
        header['history'] = i
    hdul = fits.HDUList(fits.PrimaryHDU(data, header=header))

    if hdu_uncertainty and not frame._unct.empty:
        uncert = frame.uncertainty
        uncert_h = fits.Header()
        uncert_h[unit_key] = unit_string
        hdul.append(fits.ImageHDU(uncert, header=uncert_h,
                                  name=hdu_uncertainty))

    if hdu_mask is not None and not frame._mask.empty:
        mask = frame.mask
        if np.issubdtype(mask.dtype, np.bool):
            # Fits do not support bool
            mask = mask.astype('uint8')
        hdul.append(fits.ImageHDU(mask, name=hdu_mask))

    return hdul


def _write_fits(frame, filename, overwrite=True, **kwargs):
    """Write a framedata to a fits file."""
    # FIXME: electron unit is not compatible with fits standards
    with warnings.catch_warnings():
        # silent hierarch warnings
        warnings.filterwarnings("ignore", category=VerifyWarning)
        frame.to_hdu(**kwargs).writeto(filename, overwrite=overwrite,
                                       output_verify='silentfix')
