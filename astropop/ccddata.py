# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Custom CCDData class to support memmapping."""

import os
import shutil
import numpy as np
from astropy.nddata.ccddata import CCDData as AsCCDData
from astropy.nddata.nduncertainty import NDUncertainty
from tempfile import mkdtemp, mkstemp

from .py_utils import mkdir_p


__all__ = ['CCDData', 'enable_data_memmap', 'disable_data_memmap',
           'enable_mask_memmap', 'disable_mask_memmap',
           'enable_uncertainty_memmap', 'disable_uncertainty_memmap',
           'create_array_memmap', 'delete_array_memmap']


def create_array_memmap(filename, data, dtype=None):
    """Create a memory map to an array data."""
    if data is None:
        return
    dtype = dtype or data.dtype
    shape = data.shape
    if data.ndim > 0:
        memmap = np.memmap(filename, mode='w+', dtype=dtype, shape=shape)
        memmap[:] = data[:]
    else:
        memmap = data
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
    if hasattr(ccddata, 'cache_folder'):
        cache_folder_ccd = ccddata.cache_folder
    elif hasattr(ccddata, 'parent_nddata'):
        # NDUncertainty
        if hasattr(ccddata.parent_nddata, 'cache_folder'):
            cache_folder_ccd = ccddata.parent_nddata.cache_folder
        else:
            cache_folder_ccd = None
    else:
        cache_folder_ccd = None

    cache_folder = cache_folder_ccd or cache_folder
    cache_folder = cache_folder or mkdtemp(prefix='astropop')

    if hasattr(ccddata, 'cache_filename'):
        filename_ccd = ccddata.cache_filename
    elif hasattr(ccddata, 'parent_nddata'):
        # NDUncertainty
        if hasattr(ccddata.parent_nddata, 'cache_filename'):
            filename_ccd = ccddata.parent_nddata.cache_filename
        else:
            filename_ccd = None
    else:
        filename_ccd = None

    filename = filename_ccd or filename
    filename = filename or mkstemp(suffix='.npy')[1]
    filename = os.path.basename(filename)

    ccddata.cache_folder = cache_folder
    ccddata.cache_filename = filename

    mkdir_p(cache_folder)
    return os.path.join(cache_folder, filename)


def _del_func(cls):
    # Ensure cleaning up
    del cls.data
    del cls.maks
    del cls.uncertainty


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
    cache_file = setup_filename(ccddata, cache_folder, filename)
    ccddata._mask = create_array_memmap(cache_file + '.mask', ccddata._mask)

    def _setter(self, value):
        if self._mask is None:
            name = setup_filename(self, cache_folder, filename) + '.mask'
            self._mask = create_array_memmap(name, value, dtype=bool)
        elif isinstance(self._mask, np.memmap):
            if self._mask.shape != value.shape:
                name = self._mask.filename
                delete_array_memmap(self._mask)
                self._mask = create_array_memmap(name, value, dtype=bool)
            else:
                self._mask[:] = value[:]
        else:
            delete_array_memmap(self._mask, read=False)
            self._mask = value

    def _deleter(self):
        if isinstance(self._mask, np.memmap):
            name = self._mask.filename
            dirname = os.path.dirname(name)
            del self._mask
            os.remove(name)

            if len(os.listdir(dirname)) == 0:
                shutil.rmtree(dirname)
        else:
            del self._mask

    def _getter(self):
        return self._mask

    setattr(ccddata.__class__, 'mask',
            property(fget=_getter, fset=_setter,
                     fdel=_deleter))


def disable_mask_memmap(ccddata):
    """Disable memmap caching for CCDData mask property."""
    if isinstance(ccddata._mask, np.memmap):
        ccddata._mask = delete_array_memmap(ccddata._mask)

    setattr(ccddata.__class__, 'mask', CCDData.mask)


def enable_uncertainty_memmap(ccddata, dtype=None, cache_folder=None,
                              filename=None):
    """Enable memmap caching for CCDData uncertainty property."""
    cache_file = setup_filename(ccddata, cache_folder, filename)
    if ccddata.uncertainty is not None:
        cached = create_array_memmap(cache_file + '.uncert',
                                     ccddata._uncertainty._array)
        ccddata._uncertainty._array = cached

    def _getter(self):
        return self._array

    def _setter(self, value):
        if self._array is None:
            fname = setup_filename(self, cache_folder, filename)
            self._array = create_array_memmap(fname, value)
        elif value is None:
            self._array = value
        elif isinstance(self._array, np.memmap):
            if self._array.shape != value.shape or \
               self._array.dtype != value.dtype:
                fname = self._array.filename
                delete_array_memmap(self._array)
                self._array = create_array_memmap(fname, value)
            else:
                self._array[:] = value[:]
        else:
            self._array = value

    def _deleter(self):
        if isinstance(self._array, np.memmap):
            name = self._array.filename
            dirname = os.path.dirname(name)
            del self._array
            os.remove(name)

            if len(os.listdir(dirname)) == 0:
                shutil.rmtree(dirname)
        else:
            del self._array

    setattr(ccddata.__class__, 'array',
            property(fget=_getter, fset=_setter,
                     fdel=_deleter))


def disable_uncertainty_memmap(ccddata):
    """Disable memmap caching for CCDData uncertainty property."""
    # FIXME: handle none _uncertainty
    if isinstance(ccddata._uncertainty._array, np.memmap):
        cached = delete_array_memmap(ccddata._uncertainty._array)
        ccddata._uncertainty._array = cached

    setattr(ccddata.uncertainty.__class__, 'array',
            NDUncertainty.array)


class CCDData(AsCCDData):
    """Modified astropy's CCDData to handle memmapping data from disk."""
    def __init__(self, *args, **kwargs):
        self.cache_folder = kwargs.pop('cache_folder', None)
        self.cache_filename = kwargs.pop('cache_filename', None)
        memmap = kwargs.pop('use_memmap_backend', None)
        super().__init__(*args, **kwargs)

        self._memmapping = False

        self.filename = None

        if memmap:
            self.enable_memmap()

    def enable_memmap(self, filename=None, cache_folder=None):
        """Enable array file memmapping."""
        # If is not memmapping already, create memmap
        self._memmapping = True

        enable_data_memmap(self, cache_folder=cache_folder, filename=filename)
        enable_mask_memmap(self, cache_folder=cache_folder, filename=filename)
        enable_uncertainty_memmap(self, cache_folder=cache_folder,
                                  filename=filename)

    def disable_memmap(self):
        """Disable CCDData file memmapping (load to memory)."""
        disable_data_memmap(self)
        disable_mask_memmap(self)
        disable_uncertainty_memmap(self)
        self._memmapping = False

    def __del__(self):
        # Ensure data cleaning
        del self.data
        del self.mask
        del self.uncertainty
