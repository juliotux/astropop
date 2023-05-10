# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to handle FITS files."""

from astropy.io import fits


__all__ = ['imhdus']


_fits_extensions = ['.fits', '.fts', '.fit', '.fz']
_fits_compression = ['.gz', '.zip', '.bz2', '.Z']
_fits_extensions_with_compress = _fits_extensions.copy()

for k in _fits_compression:
    _fits_extensions_with_compress.extend(
        [i+k for i in _fits_extensions_with_compress]
    )


imhdus = (fits.ImageHDU, fits.PrimaryHDU, fits.CompImageHDU,
          fits.StreamingHDU)
