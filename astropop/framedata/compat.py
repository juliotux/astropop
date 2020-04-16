# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Handle compatibility between FrameData and other data formats."""

import itertools
from astropy.io import fits
from astropy.wcs import WCS


from ..logger import logger


__all__ = ['_unsupport_fits_open_keywords', 'imhdus', 'EmptyDataError',
           'extract_header_wcs']


_unsupport_fits_open_keywords = {
    'do_not_scale_image_data': 'Image data must be scaled.',
    'scale_back': 'Scale information is not preserved.'
}


imhdus = (fits.ImageHDU, fits.PrimaryHDU, fits.CompImageHDU,
          fits.StreamingHDU)


_PCs = set(['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'])
_CDs = set(['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'])
_KEEP = set(['JD-OBS', 'MJD-OBS', 'DATE-OBS'])


def extract_header_wcs(header, logger=logger):
    """Get a header (or dict) and extract a WCS based on the keys.

    Parameters
    ----------
    header : `~astropy.fits.Header` or dict_like
        Header to extract the WCS.
    logger : `~logging.Logger` (optional)
        Logger instance to log warnings.

    Returns
    -------
    header : `~astropy.fits.Header`
        Header cleaned from WCS keys.
    wcs : `~astropy.wcs.WCS` os `None`
        World Coordinate Sistem extracted from the header.    
    """
    header = header.copy()  # Ensure original header will not be modified

    # First, check if there is a WCS. If not, return header and None WCS
    try:
        wcs = WCS(header, relax=True)
        if not wcs.wcs.ctype[0]:
            wcs = None
    except Exception as e:
        logger.warning('An error raised when extracting WCS: %s', e)
        wcs = None

    if wcs is not None:
        # Delete wcs keys, except time ones
        for k in wcs.to_header(relax=True).keys():
            if k not in _KEEP:
                header.remove(k, ignore_missing=True)

        # Astropy uses PC. So remove CD if any
        # The reverse case should not be allowed by astropy
        if (_PCs & set(wcs.to_header(relax=True))) and (_CDs & set(header)):
            for k in _CDs:
                header.remove(k, ignore_missing=True)

        # Check and remove remaining SIP coefficients
        if wcs.sip is not None:
            kwd = '{}_{}_{}'
            pol = ['A', 'B', 'AP', 'BP']
            for poly in pol:
                order = wcs.sip.__getattribute__(f'{poly.lower()}_order')
                for i, j in itertools.product(range(order), repeat=2):
                    header.remove(kwd.format(poly, i, j),
                                  ignore_missing=True)
                header.remove(f'{poly}_ORDER', ignore_missing=True)
                header.remove(f'{poly}_DMAX', ignore_missing=True)

    return (header, wcs)


class EmptyDataError(ValueError):
    """Error raised when try to operate things not handled
    by empty MemMapArray containers."""
