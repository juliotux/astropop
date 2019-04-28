# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Custom CCDData class to support memmapping."""

import os
import shutil
import numpy as np
from astropy.nddata.ccddata import CCDData as AsCCDData
from tempfile import mkdtemp, mkstemp

from .py_utils import mkdir_p


__all__ = ['CCDData']


class CCDData(AsCCDData):
    """Modified astropy's CCDData to handle memmapping data from disk."""
    def __init__(self, *args, **kwargs):
        self._cache = kwargs.pop('chace_folder', None)
        memmap = kwargs.pop('memmap', None)
        super().__init__(*args, **kwargs)

        if memmap:
            self.enable_memmap()

    @property
    def memmapping(self):
        return isinstance(self._data, np.memmap)

    def _create_memmap(self, filename, data):
        dtype = data.dtype
        shape = data.shape
        memmap = np.memmap(filename, mode='w+', dtype=dtype, shape=shape)
        memmap[:] = data[:]
        return memmap

    def _delete_memmap(self, memmap):
        data = np.array(memmap[:])
        name = memmap.filename
        del memmap
        os.remove(name)
        return data

    def enable_memmap(self, filename=None, cache_folder=None):
        """Enable CCDData file memmapping."""
        if isinstance(self._data, np.memmap):
            return

        # If is not memmapping already, create memmap
        cache_folder = cache_folder or self._cache
        cache_folder = cache_folder or mkdtemp(prefix='astropop')
        mkdir_p(cache_folder)
        if filename is None:
            f = mkstemp(prefix='ccddata', suffix='.npy',
                        dir=cache_folder)
            filename = f[1]
        else:
            filename = os.path.join(cache_folder,
                                    os.path.basename(filename))

        self._data = self._create_memmap(filename, self._data)

    def disable_memmap(self):
        """Disable CCDData file memmapping (load to memory)."""
        if isinstance(self._data, np.memmap):
            self._data = self._delete_memmap(self._data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if self.memmaping:
            if self._data.shape != value.shape or \
               self._data.dtype != value.dtype:
                name = self._data.filename
                self._delete_memmap(self._data)
                self._create_memmap(name, value)
            else:
                self._data[:] = value[:]
        else:
            self._data = value

    def __del__(self):
        if self.memmapping:
            dirname = os.path.dirname(self._data.filename)
            self._delete_memmap(self._data)

            if len(os.listdir(dirname)) == 0:
                shutil.rmtree(dirname)
