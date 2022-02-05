# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to handle FITS files."""

from astropy.io import fits
from .logger import logger


__all__ = ['imhdus', 'check_header_keys']


_fits_extensions = ['.fits', '.fts', '.fit', '.fz']
_fits_compression = ['.gz', '.zip', '.bz2', '.Z']
_fits_extensions_with_compress = _fits_extensions.copy()

for k in _fits_compression:
    _fits_extensions_with_compress.extend(
        [i+k for i in _fits_extensions_with_compress]
    )


imhdus = (fits.ImageHDU, fits.PrimaryHDU, fits.CompImageHDU,
          fits.StreamingHDU)


class IncompatibleHeadersError(ValueError):
    """When 2 image header are not compatible."""


def check_header_keys(image1, image2, keywords=None):
    """Compare header keys from 2 images to check if the have equal values."""
    keywords = keywords or []
    header1 = {}
    header2 = {}

    # Compatibility with fits HDU and FrameData
    if hasattr(image1, 'header'):
        header1 = image1.header
    elif hasattr(image1, 'meta'):
        header1 = image1.meta

    if hasattr(image2, 'header'):
        header2 = image2.header
    elif hasattr(image2, 'meta'):
        header2 = image2.meta

    for i in keywords:
        if i in header1 and i in header2:
            v1 = header1[i]
            v2 = header2[i]
            if v1 != v2:
                raise IncompatibleHeadersError(f'Keyword `{i}` have different '
                                               'values for images 1 and 2:'
                                               '`{v1}`  `{v2}`')
        elif i in header1 or i in header2:
            raise IncompatibleHeadersError("Headers have inconsisten presence "
                                           f"of {i} Keyword")
        else:
            logger.debug("The images do not have the %s keyword", i)
    return True
