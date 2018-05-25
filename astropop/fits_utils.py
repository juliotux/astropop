# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import six
import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack
from collections import OrderedDict
import functools

from .py_utils import check_iterable, process_list
from .logger import logger

__all__ = ['imhdus', 'check_header_keys', 'check_hdu', 'fits_yielder',
           'headers_to_table']

imhdus = (fits.ImageHDU, fits.PrimaryHDU, fits.CompImageHDU,
          fits.StreamingHDU)


class IncompatibleHeadersError(ValueError):
    """When 2 image header are not compatible."""


def check_header_keys(image1, image2, keywords=[]):
    """Compare header keys from 2 images to check if the have equal values."""
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


def check_hdu(data, ext=0):
    """Check if a data is a valid ImageHDU type and convert it."""
    if not isinstance(data, imhdus):
        if isinstance(data, fits.HDUList):
            data = data[0]
        elif isinstance(data, six.string_types):
            data = fits.open(data)[ext]
        elif isinstance(data, np.ndarray):
            data = fits.PrimaryHDU(data)
        else:
            raise ValueError('The given data is not a valid CCDData type.')
    return data


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
        path to save a copy of the files, with the modified object
        (header, data, hdu)
    overwrite : bool
        If overwrite existing files.
    """
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
        func = functools.partial(check_hdu, ext=ext)
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

        hdul.writeto(save_fname, overwrite=overwrite)

    for i in file_list:

        obj = func(i)
        yield obj

        if save_to:
            basename = os.path.basename(i)
            if append_to_name is not None:
                base, extf = os.path.splitext(basename)
                basename = "{}{}.{}".format(base, append_to_name, extf)

            base, extf = os.path.splitext(basename)
            if extf not in ['fits', 'fts', 'fit', 'gz', 'bz2', 'fz']:
                logger.warn('{} extension not supported for writing. '
                            'Changing to fits'.format(extf))
                subext = os.path.splitext(base)[1]
                if subext in ['fits', 'fts', 'fit', 'fz']:
                    nextf = ''
                else:
                    nextf = 'fits'
                basename = "{}.{}".format(base, nextf)

            save_fname = os.path.join(save_to, basename)

            _save(i, save_fname, obj)


def headers_to_table(headers, filenames=None, keywords=None, empty_value=None,
                     lower_keywords=False):
    """Read a bunch of headers and return a table with the values."""
    l = list(headers)
    n =len(l)

    if keywords is None or keywords == '*' or keywords == 'all':
        keywords = []
        for head in l:
            for k in head.keys():
                if k not in keywords:
                    keywords.append(k.lower() if lower_keywords else k)

    # Clean history and comment keywords. file wont be copied
    keywords = [k for k in keywords if k.lower() not in ('history', 'comment',
                                                         '', 'file')]

    headict = OrderedDict()
    for k in keywords:
        headict[k] = [empty_value]*n

    for i in range(n):
        for key, val in l[i].items():
            key = key.lower()
            if key in keywords:
                headict[key][i] = val

    if n == 0:
        if len(keywords) > 0:
            t = Table(names=keywords)
        else:
            t = Table()
    else:
        t = Table(headict, masked=True)

    if check_iterable(filenames):
        c = Table()
        c['file'] = process_list(os.path.basename, filenames)
        t = hstack([c, t])

    for k in keywords:
        t[k].mask = [v is empty_value for v in headict[k]]
    return t
