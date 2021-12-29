# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from astropy.io import fits
from astropy.io.fits.hdu.base import _ValidHDU
import functools

from .framedata import imhdus
from .logger import logger

__all__ = ['imhdus', 'check_header_keys', 'fits_yielder']


_fits_extensions = ['.fits', '.fts', '.fit', '.fz']
_fits_compression = ['.gz', '.zip', '.bz2', '.Z']
_fits_extensions_with_compress = _fits_extensions.copy()

for k in _fits_compression:
    _fits_extensions_with_compress.extend(
        [i+k for i in _fits_extensions_with_compress]
    )


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


def fits_yielder(return_type, file_list, ext=0, append_to_name=None,
                 save_to=None, overwrite=True):
    """Create a generator object that iterates over file_list.

    return_type : str
        'header', 'data' or 'hdu'
    file_list : list-like
        list of file names to be iterated
    ext : int or str
        fits extension to load the data
    append_to_name : str
        string to be appended to the file name when saving the new object
    save_to : str
        path to save a copy of the files with the modified object
        (header, data, hdu)
    overwrite : bool
        If overwrite existing files.
    """
    def _read(f):
        """Read just one hdu."""
        return fits.open(f)[ext]

    if save_to:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        elif not os.path.isdir(save_to):
            raise ValueError('Saving location is not a valid directory!')

    if return_type == 'header':
        func = functools.partial(fits.getheader, ext=ext)
    elif return_type == 'data':
        func = functools.partial(fits.getdata, ext=ext)
    elif return_type == 'hdu':
        func = _read
    else:
        raise ValueError(f'Generator {return_type} not recognized.')

    # if the image list contain hdus, re-yield them
    def _reyield(ver_obj):
        ret = None
        if return_type == 'header':
            ret = ver_obj.header
        elif return_type == 'data':
            ret = ver_obj.data
        elif return_type == 'hdu':
            ret = ver_obj
        return ret

    def _save(old, new, yielded):
        hdul = fits.open(old)
        index = hdul.index_of(ext)
        if return_type == 'header':
            hdul[index].header = yielded
        elif return_type == 'data':
            hdul[index].data = yielded
        elif return_type == 'hdu':
            hdul[index] = yielded

        hdul.writeto(new, overwrite=overwrite)

    for i in file_list:
        if isinstance(i, _ValidHDU):
            obj = _reyield(i)
        else:
            obj = func(i)

        yield obj

        if save_to:
            basename = os.path.basename(i)
            if append_to_name is not None:
                base, extf = os.path.splitext(basename)
                basename = f"{base}{append_to_name}.{extf}"

            base, extf = os.path.splitext(basename)
            if extf not in ['fits', 'fts', 'fit', 'gz', 'bz2', 'fz']:
                logger.warning('%s extension not supported for writing. '
                               'Changing to fits', str(extf))
                subext = os.path.splitext(base)[1]
                if subext in ['fits', 'fts', 'fit', 'fz']:
                    nextf = ''
                else:
                    nextf = 'fits'
                basename = f"{base}.{nextf}"

            save_fname = os.path.join(save_to, basename)

            _save(i, save_fname, obj)
