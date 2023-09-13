# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Handle compatibility between FrameData and other data formats."""

import itertools
import warnings
from os import PathLike
import copy as cp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
from astropy.nddata.ccddata import CCDData
from astropy.nddata import StdDevUncertainty, VarianceUncertainty, \
                           InverseVariance
from astropy import units as u


from ..logger import logger
from ..fits_utils import imhdus, string_to_header_key


_HDU_UNCERT = 'UNCERT'
_HDU_MASK = 'MASK'
_HDU_FLAGS = 'PIXFLAGS'
_UNIT_KEY = 'BUNIT'
_PCs = set(['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'])
_CDs = set(['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'])
_KEEP = set(['JD-OBS', 'MJD-OBS', 'DATE-OBS'])
_PROCTECTED = ["SIMPLE", "XTENSION", "BITPIX", "NAXIS", "EXTEND", "PCOUNT",
               "GCOUNT", "GROUPS", "BSCALE", "BZERO", "TFIELDS"]
_PROTECTED_N = ["TFORM", "TSCAL", "TZERO", "TNULL", "TTYPE", "TUNIT", "TDISP",
                "TDIM", "THEAP", "TBCOL"]


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


def _normalize_and_strip_dict(meta):
    """Normalize meta keys and remove the protected ones."""
    if meta is None:
        return fits.Header(), [], []

    if not isinstance(meta, fits.Header):
        raise TypeError('meta must be a fits.Header instance.')

    nmeta = fits.Header()
    history = []
    comment = []

    for k, v in meta.items():
        k = string_to_header_key(k)
        # pop history and comment
        if k == 'HISTORY':
            if np.isscalar(v):
                history.append(v)
            else:
                history.extend(v)
            continue
        if k == 'COMMENT':
            if np.isscalar(v):
                comment.append(v)
            else:
                comment.extend(v)
            continue

        if k in nmeta:
            # as dicts are case sensitive, this can happen. Raise error
            warnings.warn(f'Duplicated key {k}. First value will be used.')
        if k not in _PROCTECTED and k != '' and k not in nmeta:
            # keep commentaries. Do not keep protected keys
            nmeta.append((k, meta[k], meta.comments[k]))

    # remove protected keys with number
    naxis = nmeta.get('NAXIS', 0)
    tfields = nmeta.get('TFIELDS', 0)
    for i in range(1, naxis+1):
        if f'NAXIS{i}' in meta:
            meta.pop(f'NAXIS{i}')
    for i in range(1, tfields+1):
        for k in _PROTECTED_N:
            if f'{k}{i}' in meta:
                meta.pop(f'{k}{i}')

    return nmeta, history, comment


def _extract_fits(obj, hdu=0, unit=None, hdu_uncertainty=_HDU_UNCERT,
                  hdu_flags=_HDU_FLAGS, hdu_mask=_HDU_MASK,
                  unit_key=_UNIT_KEY):
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

    # TODO: extract flags

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
    if uncertainty is not None:
        uncertainty = StdDevUncertainty(uncertainty, unit=unit)
    mask = np.array(frame.mask)

    return CCDData(data, unit=unit, meta=meta, wcs=wcs,
                   uncertainty=uncertainty, mask=mask)


def _to_hdu(frame, hdu_uncertainty=_HDU_UNCERT, hdu_flags=_HDU_FLAGS,
            hdu_mask=_HDU_MASK, unit_key=_UNIT_KEY, wcs_relax=True, **kwargs):
    """Translate a FrameData to an HDUList."""
    data = frame.data.copy()

    # Clean header
    header = fits.Header(frame.header)

    if frame.wcs is not None:
        header.extend(frame.wcs.to_header(relax=wcs_relax),
                      update=True)

    # using bare string avoids astropy complaining about electrons
    no_fits_std = kwargs.pop('no_fits_standard_units', True)
    if no_fits_std:
        unit_string = frame.unit.to_string()
    else:
        unit_string = u.format.Fits.to_string(frame.unit)
    header[unit_key] = unit_string

    if frame.history is not None:
        for i in frame.history:
            header['history'] = i
    if frame.comment is not None:
        for i in frame.comment:
            header['comment'] = i
    hdul = fits.HDUList(fits.PrimaryHDU(data, header=header))

    if hdu_uncertainty and frame._unct is not None:
        uncert = frame.uncertainty
        uncert_h = fits.Header()
        uncert_h[unit_key] = unit_string
        hdul.append(fits.ImageHDU(uncert, header=uncert_h,
                                  name=hdu_uncertainty))

    if hdu_mask:
        mask = frame.mask.astype(np.uint8)
        if mask is not None:
            mask_h = fits.Header()
            hdul.append(fits.ImageHDU(mask, header=mask_h, name=hdu_mask))

    # TODO: add flags

    return hdul


def _write_fits(frame, filename, overwrite=True, **kwargs):
    """Write a framedata to a fits file."""
    with warnings.catch_warnings():
        # silent hierarch warnings
        warnings.filterwarnings("ignore", category=VerifyWarning)
        frame.to_hdu(**kwargs).writeto(filename, overwrite=overwrite,
                                       output_verify='silentfix')
