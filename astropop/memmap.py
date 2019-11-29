# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dynamic memmap arrays, that can be enabled or disabled."""

import os
import numpy as np

from .py_utils import check_iterable


__all__ = ['MemMapArray', 'create_array_memmap', 'delete_array_memmap']


array_bi = [f'__{i}__' for i in
             ['lt', 'le', 'gt', 'ge', 'eq', 'ne', 'bool',
              'neg', 'pos', 'abs', 'invert', 'matmul',
              'add', 'sub', 'mul', 'truediv', 'floordiv',
              'mod', 'divmod', 'pow', 'lshift', 'rshift',
              'and', 'or', 'xor',
              'iadd', 'isub', 'imul', 'itruediv', 'ifloordiv',
              'imod', 'ipow', 'ilshift', 'irshift',
              'iand', 'ior', 'ixor',
              'len', 'contains',  # 'array',
              'int', 'float', 'complex']]

array_attr = ['flags', 'shape', 'strides', 'ndim', 'data', 'size',
              'itemsize', 'nbytes', 'base', 'dtype', 'T', 'real',
              'imag', 'flat', 'ctypes', 'item', 'tolist', 'itemset',
              'tostring', 'tobytes', 'tofile', 'dump', 'dumps', 'astype',
              'byteswap', 'copy', 'view', 'getfield', 'setflags',
              'reshape', 'resize', 'transpose', 'swapaxes', 'flatten',
              'ravel', 'squeeze', 'take', 'put', 'repeat', 'choose', 'sort',
              'argsort', 'partition', 'argpartition', 'searchsorted', 'nonzero',
              'compress', 'diagonal', 'max', 'argmax', 'min', 'argmin', 'ptp',
              'conj', 'round', 'trace', 'sum', 'cumsum', 'mean', 'var', 'std',
              'prod', 'cumprod', 'all', 'any']

redirects = array_attr + array_bi


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


def delete_array_memmap(memmap, read=True, remove=False):
    """Delete a memmap and read the data to a np.ndarray"""
    if memmap is None:
        return

    if read:
        data = np.array(memmap[:])
    else:
        data = None
    name = memmap.filename
    if remove:
        del memmap
        os.remove(name)
    return data


def to_memmap_attr(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return MemMapArray(result)
        return result
    return wrapper


class MemMapArray:
    # TODO: __copy__
    _filename = None  # filename of memmap
    _file_lock = False  # lock filename
    _contained = None  # Data contained: numpy ndarray or memmap

    def __init__(self, data, filename=None, dtype=None, unit=None):
        # None data should generate a empty container
        if data is None:
            self._contained = None
        else:
            if hasattr(data, 'dtype'):
                dtype = dtype or data.dtype
            else:
                data = np.array(data)
                dtype = dtype or data.dtype
            self._contained = np.array(data, dtype=dtype or 'float64')  # Default dtype
        self.set_filename(filename)
        self.set_unit(unit)

    @property  # read only
    def empty(self):
        return self._contained is None
    
    @property  # read only
    def unit(self):
        return self._unit

    @property  # read only
    def filename(self):
        if self.memmap:
            return self._contained.filename
        return self._filename

    @property  # read unly
    def memmap(self):
        """True if memmap is enabled."""
        return self._memmap

    def set_filename(self, value):
        """Set the memmap filename.
        
        If the instance is already memmapping, memmap will be moved to the new file.

        Parameters:
        -----------
            value : string
                New memmap filename.
        """
        if not self._file_lock:
            self._filename = value
        elif value != self._filename:
            raise ValueError('Filename locked.')

        if self.memmap:
            if value != self._contained.filename:
                n_mm = create_array_memmap(value, self._contained)
                delete_array_memmap(self._contained, read=False, remove=True)
                self._contained = n_mm

    def set_unit(self, value=None):
        """Set the data physical unit.
        
        Parameters:
        -----------
            value = string or `astropy.units.Unit`
                New physical unit of the data.
        """
        from astropy import units as u

        if value is None:
            self._unit = u.dimensionless_unscaled
        else:
            self._unit = u.Unit(value) 

    def enable_memmap(self, filename=None):
        """Enable data file memmapping (write data to disk).
        
        Parameters:
        -----------
            filename : string or None (optional)
                File name of new memmapping. If `None`, the class default value will
                be used.  
        """
        if self.memmap:
            return
        
        if filename is not None:
            self.set_filename(filename)

        self._contained = create_array_memmap(self._filename, self._contained)
        self._memmap = True

    def disable_memmap(self, remove=False):
        """Disable data file memmapping (read to memory).
        
        Parameters:
        -----------
            remove : bool
                Remove the memmap file after read values.
        """
        if not self.memmap:
            return
        
        self._contained = delete_array_memmap(self._contained, read=True, remove=remove)
        self._memmap = False

    def flush(self):
        """Write changes to disk if memmapping."""
        if self.memmap:
            self._contained.flush()
    
    def reset_data(self, data=None, unit=None, dtype=None):
        """Set new data.
        
        Parameters:
        -----------
            data : np.ndarray or None (optional)
                Data array to be owned. If None, the container will be empty.
            unit : string or `astropy.units.Unit` (optional)
                Physical unit of the data.
            dtype : string or `numpy.dtype` (optional)
                Imposed data type.
        """
        if data is None and self.memmap:
            mm = self._contained
            self._contained = None
            delete_array_memmap(mm, read=False, remove=True)
        elif data is None:
            self._contained = None
        elif self.memmap:
            name = self.filename
            mm = self._contained
            self._contained = create_array_memmap(name, data, dtype)
            delete_array_memmap(mm, read=False, remove=True)
        else:
            self._contained = np.array(data, dtype=dtype)

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

    def __getattribute__(self, item):
        if item in redirects and not self.empty:
            attr = getattr(self._contained, item)
            if callable(attr):
                attr = to_memmap_attr(attr)
            elif isinstance(attr, np.ndarray):
                attr = MemMapArray(attr)
            return attr
        elif item in redirects and self.empty:
            raise KeyError('Empty data container')
    
        return object.__getattribute__(self, item)
    
    def __repr__(self):
        return 'MemMapArray:\n' + repr(self._contained)

    def __array__(self):
        if self.empty:
            return np.array(None)
        elif self.memmap:
            return self._contained
        else:
            return np.array(self._contained)
