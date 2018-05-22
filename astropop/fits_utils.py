# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import six
import numpy as np
from astropy.io import fits
import functools

__all__ = ['imhdus', 'check_header_keys', 'check_hdu']

imhdus = (fits.ImageHDU, fits.PrimaryHDU, fits.CompImageHDU,
          fits.StreamingHDU)


class IncompatibleHeadersError(ValueError):
    """When 2 image header are not compatible"""


def check_header_keys(image1, image2, keywords=[]):
    """Compare some header keys from 2 images and check if the have equal
    values."""
    image1 = check_hdu(image1)
    image2 = check_hdu(image2)
    for i in keywords:
        if i in image1.header.keys():
            if i in image2.header.keys():
                v1 = image1.header[i]
                v2 = image2.header[i]
                if v1 != v2:
                    raise IncompatibleHeadersError('Keyword {} have different '
                                                   'values for images 1 and 2:'
                                                   '\n{}\n{}'.format(i,
                                                                     v1, v2))
            else:
                raise IncompatibleHeadersError('Image 2 do not have Keyword {}'
                                               .format(i))
        else:
            raise IncompatibleHeadersError('Image 1 do not have Keyword {}'
                                           .format(i))
    return True


def check_hdu(data):
    """Check if a data is a valid ImageHDU type and convert it."""
    if not isinstance(data, imhdus):
        if isinstance(data, fits.HDUList):
            data = data[0]
        elif isinstance(data, six.string_types):
            data = fits.open(data)[0]
        elif isinstance(data, np.ndarray):
            data = fits.PrimaryHDU(data)
        else:
            raise ValueError('The given data is not a valid CCDData type.')
    return data


def fits_yielder(return_type, file_list, ext=0, append_to_name=None,
                 save_to=None, overwrite=True):
    """Create a generator object of the file_list."""
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
        func = functools.partial(check_hdu, default_hdu=ext)
    else:
        raise ValueError('Generator not recognized.')

    def _save(old, new, yielded):
        hdul = fits.open(old)
        index = hdul.index_of(ext)
        if return_type == 'header':
            hdul[index].header = yielded
        elif return_type == 'data':
            hdul[index].data = yielded
        elif return_type == 'hdu':
            hdul[index] = yielded

        hdul.writeto(save_fname, overwrite=True)

    for i in file_list:
        obj = func(i)
        yield obj

        if save_to:
            basename = os.path.basename(i)
            if append_to_name:
                base, ext = os.path.splitext(basename)
                basename = base + append_to_name + ext

            save_fname = os.path.join(save_to, basename)
            _save(i, save_fname, obj)
