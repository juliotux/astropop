# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dynamic memmap arrays, that can be enabled or disabled."""

import numpy as np

from .py_utils import check_iterable


__all__ = ['MemMapArray']


def to_memmap_attr(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return MemMapArray(result)
        return result
    return wrapper


class MemMapArray:
    _memmap = False  # if the array is memmaped
    _filename = None  # filename of memmap
    _file_lock = False  # lock filename
    _contained = None  # Data contained: numpy ndarray or memmap
    _dtype = None #

    def __init__(self, data, filename=None, dtype=None):
        if data is None:
            self._contained = None
        else:
            self._contained = np.array(data)
        self._dtype = dtype
        self._filename = filename
        self._file_lock = True

    @property
    def empty(self):
        return self._contained is None

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if not self._file_lock:
            self._filename = value
        else:
            raise ValueError('Filename locked.')

    @property
    def memmap(self):
        return self._memmap

    def enable_memmap(self):
        raise NotImplementedError

    def disable_memmap(self):
        raise NotImplementedError

    def __getitem__(self, item):
        if self.empty:
            raise KeyError('Empty data contaier')
        result = self._contained[item]
        if isinstance(result, np.ndarray):
            result = MemMapArray(result, self._dtype or result.dtype)
        return result

    def __setitem__(self, item, value):
        if self.empty:
            raise KeyError('Empty data container')
        self._contained[item] = value

    def __getattr__(self, item):
        if item in self.__dict__.keys():
            return self.__dict__[item]
        elif not self.empty:
            attr = getattr(self._contained, item)
            if callable(attr):
                attr = to_memmap_attr(attr)
            return attr
        else:
            raise AttributeError(item)
    
    def __repr__(self):
        return 'MemMapArray:\n' + repr(self._contained)
    



