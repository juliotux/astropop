# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Custom CCDData class to support memmapping."""

import os
import shutil
import numpy as np
from astropy.nddata.ccddata import CCDData as AsCCDData
from tempfile import mkdtemp, mkstemp

from .py_utils import mkdir_p


__all__ = ['CCDData']


def create_array_memmap(filename, data, dtype=None):
    """Create a memory map to an array data."""
    if data is None:
        return
    dtype = dtype or data.dtype
    shape = data.shape
    memmap = np.memmap(filename, mode='w+', dtype=dtype, shape=shape)
    memmap[:] = data[:]
    return memmap


def delete_array_memmap(memmap, read=True):
    """Delete a memmap and read the data to a np.ndarray"""
    if read:
        data = np.array(memmap[:])
    else:
        data = None
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


class CCDData(AsCCDData):
    """Modified astropy's CCDData to handle memmapping data from disk."""
    def __init__(self, *args, **kwargs):
        self.cache_folder = kwargs.pop('cache_folder', None)
        self.cache_filename = kwargs.pop('cache_filename', None)
        memmap = kwargs.pop('use_memmap_backend', None)
        super().__init__(*args, **kwargs)

        if memmap:
            self.enable_memmap()

    def enable_memmap(self, filename=None, cache_folder=None):
        """Enable array file memmapping."""
        # If is not memmapping already, create memmap
        filename = setup_filename(self, cache_folder, filename)

        if not isinstance(self._data, np.memmap):
            self._data = create_array_memmap(filename + '.data', self._data)
        if not isinstance(self._mask, np.memmap):
            self._mask = create_array_memmap(filename + '.mask', self._mask,
                                             dtype=bool)

    def disable_memmap(self):
        """Disable CCDData file memmapping (load to memory)."""
        if isinstance(self._data, np.memmap):
            self._data = delete_array_memmap(self._data)
        if isinstance(self._mask, np.memmap):
            self._mask = delete_array_memmap(self._mask)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        if self._mask is None:
            name = setup_filename(self) + '.mask'
            self._mask = create_array_memmap(name, value, dtype=bool)
        elif isinstance(self._mask, np.memmap):
            if self._mask.shape != value.shape:
                name = self._mask.filename
                delete_array_memmap(self._mask)
                self._mask = create_array_memmap(name, value, dtype=bool)
            else:
                self._mask[:] = value[:]
        else:
            self._mask = value

    @mask.deleter
    def mask(self):
        if isinstance(self._mask, np.memmap):
            delete_array_memmap(self._mask, read=False)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
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

    @data.deleter
    def data(self):
        if isinstance(self._data, np.memmap):
            delete_array_memmap(self._data, read=False)

    def _clear_cache_folder(self):
        if self.cache_folder is not None:
            if len(os.listdir(self.cache_folder)) == 0:
                shutil.rmtree(self.cache_folder)

    def __del__(self):
        # Ensure data cleaning
        del self.data
        del self.mask
        self._clear_cache_folder()
