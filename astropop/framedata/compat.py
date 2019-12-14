# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits


__all__ = ['_unsupport_fits_open_keywords', 'imhdus', 'EmptyDataError']


_unsupport_fits_open_keywords = {
    'do_not_scale_image_data': 'Image data must be scaled.',
    'scale_back': 'Scale information is not preserved.'
}


imhdus = (fits.ImageHDU, fits.PrimaryHDU, fits.CompImageHDU,
          fits.StreamingHDU)


class EmptyDataError(ValueError):
    """Error raised when try to operate things not handled
    by empty MemMapArray containers."""
