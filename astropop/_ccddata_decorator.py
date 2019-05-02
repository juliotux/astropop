# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Custom CCDData class to support memmapping."""

import os
import shutil
import numpy as np
from astropy.nddata.ccddata import CCDData
from tempfile import mkdtemp, mkstemp

from .py_utils import mkdir_p


__all__ = ['enable_data_memmap',
           'disable_data_memmap']


def create_array_memmap(filename, data):
    """Create a memory map to an array data."""
    if isinstance(data, np.memmap):
        if filename == data.filename:
            return data

    dtype = data.dtype
    shape = data.shape
    memmap = np.memmap(filename, mode='w+', dtype=dtype, shape=shape)
    if data.ndim > 0:
        memmap[:] = data[:]
    return memmap


def delete_array_memmap(memmap):
    """Delete a memmap and read the data to a np.ndarray"""
    data = np.array(memmap[:])
    name = memmap.filename
    del memmap
    os.remove(name)
    return data


def setup_filename(ccddata, cache_folder=None, filename=None):
    """Setup filename and cache folder to a CCDData"""
    if not hasattr(ccddata, 'cache_folder'):
        cache_folder_ccd = None
    else:
        cache_folder_ccd = ccddata.cache_folder

    cache_folder = cache_folder_ccd or cache_folder
    cache_folder = cache_folder or mkdtemp(prefix='astropop')

    if not hasattr(ccddata, 'cache_filename'):
        filename_ccd = None
    else:
        filename_ccd = ccddata.cache_filename

    filename = filename_ccd or filename
    filename = filename or mkstemp(suffix='.npy')[1]
    filename = os.path.basename(filename)

    ccddata.cache_folder = cache_folder
    ccddata.cache_filename = filename

    mkdir_p(cache_folder)
    return os.path.join(cache_folder, filename)


def enable_data_memmap(ccddata, dtype=None, cache_folder=None, filename=None):
    """Enable memmap caching for CCDData data property."""
    cache_file = setup_filename(ccddata, cache_folder, filename)

    ccddata._data = create_array_memmap(cache_file + '.data', ccddata._data)

    def _setter(self, value):
        if isinstance(self._data, np.memmap):
            if self._data.shape != value.shape or \
               self._data.dtype != value.dtype:
                name = self._data.filename
                delete_array_memmap(self._data)
                self._data = create_array_memmap(name, value)
            else:
                self._data[:] = value[:]
        else:
            self._data = value

    def _deleter(self):
        if isinstance(self._data, np.memmap):
            name = self._data.filename
            dirname = os.path.dirname(name)
            del self._data
            os.remove(name)

            if len(os.listdir(dirname)) == 0:
                shutil.rmtree(dirname)
        else:
            del self._data

    def _getter(self):
        return self._data

    setattr(ccddata.__class__, 'data',
            property(fget=_getter, fset=_setter,
                     fdel=_deleter))


def disable_data_memmap(ccddata):
    """Disable memmap caching for CCDData data property."""
    if isinstance(ccddata._data, np.memmap):
        ccddata._data = delete_array_memmap(ccddata._data)

    setattr(ccddata.__class__, 'data', CCDData.data)


def enable_mask_memmap(ccddata, dtype=bool, cache_folder=None, filename=None):
    """Enable memmap caching for CCDData mask property."""


def disable_mask_memmap(ccddata):
    """Disable memmap caching for CCDData mask property."""


def enable_uncertainty_memmap(ccddata, dtype=None, cache_folder=None,
                              filename=None):
    """Enable memmap caching for CCDData uncertainty property."""
    raise NotImplementedError()


def disable_uncertainty_memmap(ccddata):
    """Disable memmap caching for CCDData uncertainty property."""
    raise NotImplementedError()
