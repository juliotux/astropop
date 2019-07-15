# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Custom frame class to support memmapping."""

# Important: We reimplemented and renamed astropy's frame to better
# handle disk memmap and an easier unit/uncertainty workflow

from pathlib import Path
import os
import shutil
import numpy as np
from astropy.nddata.nduncertainty import NDUncertainty
from tempfile import mkdtemp, mkstemp

from .py_utils import mkdir_p


__all__ = ['FrameData', 'create_array_memmap', 'delete_array_memmap',
           'ensure_bool_mask', 'setup_filename']


def ensure_bool_mask(value):
    """Ensure a mask value is bool"""
    if hasattr(value, 'dtype'):
        # If bool, just return
        if np.dtype(value.dtype) is np.dtype(np.bool):
            return value
    
    # handle memmap
    if isinstance(value, np.memmap):
        filename = Path(value.filename)
        value = delete_array_memmap(value)
    else:
        filename = None

    value = np.array(value).astype('bool')

    if filename is not None:
        value = create_array_memmap(filename.open('w'), value)

    return value


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


def setup_filename(frame, cache_folder=None, filename=None):
    """Setup filename and cache folder to a frame"""
    if hasattr(frame, 'cache_folder'):
        cache_folder_ccd = frame.cache_folder
    else:
        cache_folder_ccd = None

    if hasattr(frame, 'cache_filename'):
        filename_ccd = frame.cache_filename
    else:
        filename_ccd = None

    filename = filename_ccd or filename
    filename = filename or mkstemp(suffix='.npy')[1]
    filename = os.path.basename(filename)
    if cache_folder is None and os.path.dirname(filename) != '':
        cache_folder = os.path.dirname(filename)

    cache_folder = cache_folder_ccd or cache_folder
    cache_folder = cache_folder or mkdtemp(prefix='astropop')

    frame.cache_folder = cache_folder
    frame.cache_filename = filename

    mkdir_p(cache_folder)
    return os.path.join(cache_folder, filename)


class FrameData:
    """Modified astropy's frame to handle memmapping data from disk."""
    # TODO: Uncertainty
    # TODO: Math operations (__add__, __subtract__, etc...)
    _memmapping = False
    _data = None
    _data_unit = None
    _mask = None
    _uncertainty = None
    _uncert_type = None
    _uncert_unit = None

    def __init__(self, *args, **kwargs):
        self.cache_folder = kwargs.pop('cache_folder', None)
        self.cache_filename = kwargs.pop('cache_filename', None)
        memmap = kwargs.pop('use_memmap_backend', None)

        self._memmapping = False

        self.filename = None

        if memmap:
            self.enable_memmap()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        # TODO: Handle quantities
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
            name = self._data.filename
            dirname = os.path.dirname(name)
            del self._data
            os.remove(name)

            if len(os.listdir(dirname)) == 0:
                shutil.rmtree(dirname)
        else:
            del self._data

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        if self._mask is None and self._memmapping:
            name = setup_filename(self, cache_folder, filename) + '.mask'
            self._mask = create_array_memmap(name, value, dtype=bool)
        elif self._mask is None:
            self._mask = value
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

    @mask.deleter
    def mask(self):
        if isinstance(self._mask, np.memmap):
            name = self._mask.filename
            dirname = os.path.dirname(name)
            del self._mask
            os.remove(name)

            if len(os.listdir(dirname)) == 0:
                shutil.rmtree(dirname)
        else:
            del self._mask

    def enable_memmap(self, filename=None, cache_folder=None):
        """Enable array file memmapping."""
        cache_file = setup_filename(self, cache_folder, filename)

        # Data
        self._data = create_array_memmap(cache_file + '.data', self._data)

        # Mask
        frame.mask = create_array_memmap(cache_file + '.mask', self._mask)

        self._memmapping = True

    def disable_memmap(self):
        """Disable frame file memmapping (load to memory)."""

        self._memmapping = False

    # def __del__(self):
    #     # Ensure data cleaning
    #     del self.data
    #     del self.mask
    #     del self.uncertainty
