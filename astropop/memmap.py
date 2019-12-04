# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dynamic memmap arrays, that can be enabled or disabled."""

import os
import numpy as np
from astropy import units as u

from .py_utils import check_iterable


__all__ = ['MemMapArray', 'create_array_memmap', 'delete_array_memmap']


redirects = ['flags', 'shape', 'strides', 'ndim', 'data', 'size',
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


def create_array_memmap(filename, data, dtype=None):
    """Create a memory map to an array data."""
    if data is None:
        return

    if filename is None:
        raise ValueError('Could not create a memmap file with None filename.')

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


def to_memmap_operator(item):
    def wrapper(self, *args, **kwargs):
        if not self.empty:
            func = self._contained.__getattribute__(item)
            func = to_memmap_attr(func)
            return func(*args, **kwargs)
        else:
            # TODO: Think if this is the best behavior
            return None
    return wrapper


def to_memmap_attr(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return MemMapArray(result, memmap=False)
        return result
    return wrapper


class MemMapArray:
    # TODO: __copy__
    _filename = None  # filename of memmap
    _file_lock = False  # lock filename
    _contained = None  # Data contained: numpy ndarray or memmap
    _memmap = False
    _unit = u.dimensionless_unscaled

    def __init__(self, data, filename=None, dtype=None, unit=None, memmap=True):
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
        self._file_lock = True

        if memmap:
            self.enable_memmap()

    @property  # read only
    def empty(self):
        return self._contained is None
    
    @property  # read only
    def unit(self):
        return self._unit

    @property  # read only
    def filename(self):
        if self.memmap and self._contained is not None:
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
        if data is None:
            if self.memmap:
                # Need to delete memmap
                mm = self._contained
                self._contained = None
                delete_array_memmap(mm, read=False, remove=True)
            else:
                # don't need to delete memmap
                self._contained = None
            self.set_unit(None)

        # Not None data
        else:
            if self.memmap:
                name = self.filename
                mm = self._contained
                self._contained = create_array_memmap(name, data, dtype)
                delete_array_memmap(mm, read=False, remove=True)
            else:
                self._contained = np.array(data, dtype=dtype)

            # Unit handling
            if hasattr(data, 'unit'):
                dunit = u.Unit(data.unit)
                if unit is not None:
                    unit = u.Unit(unit) 
                    if unit is not dunit:
                        raise ValueError(f'unit={unit} set for a Quantity data '
                                         f'with {dunit} unit.')
                self.set_unit(dunit)
            else:
                if unit is not None:
                    self.set_unit(unit)
                # if None, keep the old unit

    def __getitem__(self, item):
        if self.empty:
            raise KeyError('Empty data contaier')

        # This cannot create a new MemMapArray to don't break a[x][y] = z
        result = self._contained[item]
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
                attr = MemMapArray(attr, memmap=False)
            return attr
        elif item in redirects and self.empty:
            raise KeyError('Empty data container')
    
        return object.__getattribute__(self, item)
    
    def __repr__(self):
        return 'MemMapArray:\n' + repr(self._contained) + f'\nfile: {self.filename}'

    def __array__(self):
        if self.empty:
            return np.array(None)
        elif self.memmap:
            return self._contained
        else:
            return np.array(self._contained)

    __lt__ = to_memmap_operator('__lt__')
    __le__ = to_memmap_operator('__le__')
    __gt__ = to_memmap_operator('__gt__')
    __ge__ = to_memmap_operator('__ge__')
    __eq__ = to_memmap_operator('__eq__')
    __ne__ = to_memmap_operator('__ne__')
    __add__ = to_memmap_operator('__add__')
    __sub__ = to_memmap_operator('__sub__')
    __mul__ = to_memmap_operator('__mul__')
    __pow__ = to_memmap_operator('__pow__')
    __truediv__ = to_memmap_operator('__truediv__')
    __floordiv__ = to_memmap_operator('__floordiv__')
    __mod__ = to_memmap_operator('__mod__')
    __lshift__ = to_memmap_operator('__lshift__')
    __rshift__ = to_memmap_operator('__rshift__')
    __and__ = to_memmap_operator('__and__')
    __or__ = to_memmap_operator('__or__')
    __xor__ = to_memmap_operator('__xor__')
    __neg__ = to_memmap_operator('__neg__')
    __pos__ = to_memmap_operator('__pos__')
    __abs__ = to_memmap_operator('__abs__')
    __invert__ = to_memmap_operator('__invert__')
    __matmul__ = to_memmap_operator('__matmul__')
    __bool__ = to_memmap_operator('__bool__')
    __float__ = to_memmap_operator('__float__')
    __complex__ = to_memmap_operator('__complex__')
    __int__ = to_memmap_operator('__int__')
    __iadd__ = to_memmap_operator('__iadd__')
    __isub__ = to_memmap_operator('__isub__')
    __ipow__ = to_memmap_operator('__ipow__')
    __imul__ = to_memmap_operator('__imul__')
    __itruediv__ = to_memmap_operator('__itruediv__')
    __ifloordiv__ = to_memmap_operator('__ifloordiv__')
    __imod__ = to_memmap_operator('__imod__')
    __ilshift__ = to_memmap_operator('__ilshift__')
    __irshift__ = to_memmap_operator('__irshift__')
    __iand__ = to_memmap_operator('__iand__')
    __ior__ = to_memmap_operator('__ior__')
    __ixor__ = to_memmap_operator('__ixor__')
    __len__ = to_memmap_operator('__len__')
    __contains__ = to_memmap_operator('__contains__')
