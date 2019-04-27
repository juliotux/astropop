# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Custom CCDData class to support streaming."""

import os
import shutil
import numpy as np
from astropy.nddata.ccddata import CCDData as AsCCDData
from tempfile import mkdtemp, mkstemp

from .py_utils import mkdir_p


__all__ = ['CCDData']


class CCDData(AsCCDData):
    """Modified astropy's CCDData to handle streaming data from disk."""
    def __init__(self, *args, **kwargs):
        self._cache = kwargs.pop('chace_folder', None)
        stream = kwargs.pop('stream', None)
        super().__init__(*args, **kwargs)

        if stream:
            self.enable_stream()

    @property
    def streaming(self):
        return isinstance(self._data, np.memmap)

    def _create_stream(self, filename, data):
        dtype = data.dtype
        shape = data.shape
        stream = np.memmap(filename, mode='w+', dtype=dtype, shape=shape)
        stream[:] = data[:]
        return stream

    def _delete_stream(self, stream):
        data = np.array(stream[:])
        name = stream.filename
        del stream
        os.remove(name)
        return data

    def enable_stream(self, filename=None, cache_folder=None):
        """Enable CCDData file streaming."""
        if isinstance(self._data, np.memmap):
            return

        # If is not streaming already, create stream
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

        self._data = self._create_stream(filename, self._data)

    def disable_stream(self):
        """Disable CCDData file streaming (load to memory)."""
        if isinstance(self._data, np.memmap):
            self._data = self._delete_stream(self._data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if self.streaming:
            if self._data.shape != value.shape:
                name = self._data.filename
                self._delete_stream(self._data)
                self._create_stream(name, value)
            else:
                self._data[:] = value[:]
        else:
            self._data = value

    def __del__(self):
        if self.streaming:
            dirname = os.path.dirname(self._data.filename)
            self._delete_stream(self._data)

            if len(os.listdir(dirname)) == 0:
                shutil.rmtree(dirname)
