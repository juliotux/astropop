# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import six
import numpy as np
from astropy.io import fits
from astropy.io.fits.hdu.base import _ValidHDU
from astropy.nddata import NDData, StdDevUncertainty, CCDData
from astropy import units as u
from astropy.table import Table, hstack
from collections import OrderedDict
import functools

from .py_utils import check_iterable, process_list
from .framedata import FrameData, imhdus
from .logger import logger

__all__ = ['imhdus', 'check_header_keys', 'check_image_hdu', 'fits_yielder',
           'headers_to_table']


_supported_formats = [".fts", ".fit", ".fz", ".fits"]
_compresses = [".gz", ".bz2", ".zip"]
for k in _compresses:
    _supported_formats.extend([i+k for i in _supported_formats])


class IncompatibleHeadersError(ValueError):
    """When 2 image header are not compatible."""


def check_header_keys(image1, image2, keywords=[], logger=logger):
    """Compare header keys from 2 images to check if the have equal values."""
    # Compatibility with fits HDU and FrameData
    if hasattr(image1, 'header'):
        hk1 = 'header'
    elif hasattr(image1, 'meta'):
        hk1 = 'meta'
    if hasattr(image2, 'header'):
        hk2 = 'header'
    elif hasattr(image2, 'meta'):
        hk2 = 'meta'

    image1 = check_image_hdu(image1)[hk1]
    image2 = check_image_hdu(image2)[hk2]
    for i in keywords:
        if i in image1.keys() and i in image2.keys():
            v1 = image1[i]
            v2 = image2[i]
            if v1 != v2:
                raise IncompatibleHeadersError(f'Keyword `{i}` have different '
                                               'values for images 1 and 2:'
                                               '`{v1}`  `{v2}`')
        elif i in image1.header.keys() or i in image2.header.keys():
            raise IncompatibleHeadersError("Headers have inconsisten presence "
                                           f"of {i} Keyword")
        else:
            logger.debug(f"The images do not have the {i} keyword")
    return True


def check_image_hdu(data, ext=0, logger=logger):
    """Check if a data is a valid ImageHDU type or convert it."""
    if not isinstance(data, imhdus):
        if isinstance(data, fits.HDUList):
            logger.debug(f"Extracting HDU from ext {ext} of HDUList")
            data = data[ext]
        elif isinstance(data, six.string_types):
            data = fits.open(data)[ext]
        elif isinstance(data, np.ndarray):
            data = fits.PrimaryHDU(data)
        else:
            raise ValueError('The given data is not a valid CCDData type.')
    return data


def save_image_hdu(hdu, filename, overwrite=False, logger=logger):
    """Save simple HDU to a fits file."""
    base, ext = os.path.splitext(filename)
    if ext in _compresses:
        ext2 = ext
        base, ext = os.path.splitext(base)
    elif ext not in _supported_formats:
        # TODO: think its better to save fits or raise error.
        ext = ".fits"
    else:
        ext2 = None

    if ext2 is not None:
        ext += ext2

    filename = base + ext
    logger.debug(f'Saving fits file to: {filename}')

    if ext == '.fz':
        p = fits.PrimaryHDU()
        c = fits.CompImageHDU(hdu.data, header=hdu.header,
                              compression_type='RICE_1')
        fits.HDUList([p, c]).writeto(filename, overwrite=overwrite)
    else:
        hdu.writeto(filename, overwrite=overwrite)


def fits_yielder(return_type, file_list, ext=0, append_to_name=None,
                 save_to=None, overwrite=True, logger=logger):
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
        func = functools.partial(check_image_hdu, ext=ext)
    else:
        raise ValueError('Generator not recognized.')

    # if the image list contain hdus, re-yield them
    def _reyield(ver_obj):
        if return_type == 'header':
            return ver_obj.header
        elif return_type == 'data':
            return ver_obj.data
        elif return_type == 'hdu':
            return ver_obj

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
                logger.warn(f'{extf} extension not supported for writing. '
                            'Changing to fits')
                subext = os.path.splitext(base)[1]
                if subext in ['fits', 'fts', 'fit', 'fz']:
                    nextf = ''
                else:
                    nextf = 'fits'
                basename = f"{base}.{nextf}"

            save_fname = os.path.join(save_to, basename)

            _save(i, save_fname, obj)


def headers_to_table(headers, filenames=None, keywords=None, empty_value=None,
                     lower_keywords=False, logger=logger):
    """Read a bunch of headers and return a table with the values."""
    # TODO: Refactor to better performance
    hlist = []
    actual = 0
    for head in headers:
        hlist.append(head)
        actual += 1
        logger.debug(f"Reading header {actual}")

    n = len(hlist)

    if keywords is None or keywords == '*' or keywords == 'all':
        keywords = []
        logger.debug('Reading keywords.')
        for head in hlist:
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
        logger.debug(f"Processing header {i} from {n}")
        for key, val in hlist[i].items():
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
