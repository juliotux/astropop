# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dynamic memmap arrays, that can be enabled or disabled."""

import os
import numpy as np

from .compat import EmptyDataError


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
    """Create a memory map to an array data.

    Parameters
    ----------
    filename : `string`, `~pathlib.Path`
        Name of memmap file to be created.
    data : array_like
        Data to be stored in the memmap.
    dtype : `string`, `~numpy.dtype` or `None` (optional)
        `~numpy.dtype` compilant data type. If `None`, `data.dtype` will be
        used.

    Returns
    -------
    memmap : `~numpy.memmap`
        Memmap object of cached data.
    """
    if data is None:
        return None

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
    """Delete a memmap and read the data to a np.ndarray.

    Parameters
    ----------
    memmap : array_like or `None`
        MemMap array to be deleted.
    read : `bool` (optional)
        Read the data to memory before delete.
    remove : `bool` (optional)
        Delete the memmap file.

    Returns
    -------
    data : array_like or `None`
        Data read from memmap.
    """
    if memmap is None or np.shape(memmap) == ():
        return None

    if read:
        data = np.array(memmap[:])
    else:
        data = None

    if not isinstance(memmap, np.memmap):
        return data

    name = memmap.filename
    if remove:
        del memmap
        if os.path.exists(name):
            os.remove(name)
    return data


def to_memmap_operator(item):
    """Wrap operators to `MemMapArray`."""
    def wrapper(self, *args, **kwargs):
        if not self.empty:
            func = self._contained.__getattribute__(item)
            func = to_memmap_attr(func)
            return func(*args, **kwargs)
        raise EmptyDataError('Empty data container.')
    return wrapper


def to_memmap_attr(func):
    """Wrap attrs to `MemMapArray`."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return MemMapArray(result, memmap=False)
        return result
    return wrapper


class MemMapArray:
    """An array object that can be easily cached (memmaped) to disk.

    `MemMapArray` is a wrapper around `~numpy.nddata` that allows easy enable
    or disable of memmapping feature (using `~numpy.memmap`). It also wraps
    all numpy array operations and must behave like every single
    `numpy.ndarray`.

    Parameters
    ----------
    data : array_like or `None`
        The data to be assigned to array. If `None`, an empty array will be
        created.
    filename : `string`, `~pathlib.Path` or `None` (optional)
        Base file name for the cache file.
    dtype : `string`, `~numpy.dtype` or `None` (optional)
        Data type to be used in the array to be stores. If `None`, altomatic
        `numpy` dtype will be used. Must be `numpy.dtype` compilant.
    memmap : `bool` (optional)
        If the instance is set to memmap mode from the start.
        Default: `False`

    Notes
    -----
    - This is just a numeric storing array. No unit is assigned and it behaves
      like any dimensionless number in operations.
    """
    # TODO: __copy__

    _filename = None  # filename of memmap
    _file_lock = False  # lock filename
    _contained = None  # Data contained: numpy ndarray or memmap
    _memmap = False

    def __init__(self, data, filename=None, dtype=None,
                 memmap=False):
        # None data should generate a empty container
        if data is None:
            self._contained = None
        else:
            if not hasattr(data, 'dtype'):
                data = np.array(data)
            dtype = dtype or data.dtype
            # Default dtype
            dtype = dtype or 'float64'
            self._contained = np.array(data, dtype=dtype)
            # ensure native byteorder
            if not self._contained.dtype.isnative:
                self._contained = self._contained.byteswap().newbyteorder()
        self.set_filename(filename)
        self._file_lock = True

        if memmap:
            self.enable_memmap()

    @property  # read only
    def empty(self):
        """True if contained data is empty (None)."""
        return self._contained is None

    @property  # read only
    def filename(self):
        """Name of the file where data is cached, if memmap enabled."""
        if self.memmap and self._contained is not None:
            return self._contained.filename
        return self._filename

    @property  # read unly
    def memmap(self):
        """True if memmap is enabled."""
        return self._memmap

    def set_filename(self, value):
        """Set the memmap filename.

        If the instance is already memmapping, memmap will be moved to the new
        file.

        Parameters
        ----------
        value : `string`
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

    def enable_memmap(self, filename=None):
        """Enable data file memmapping (write data to disk).

        Parameters
        ----------
        filename : `string` or `None` (optional)
            File name of new memmapping. If `None`, the class default
            value will be used.
        """
        if self.memmap:
            return

        if filename is not None:
            self.set_filename(filename)

        self._contained = create_array_memmap(self._filename, self._contained)
        self._memmap = True

    def disable_memmap(self, remove=False):
        """Disable data file memmapping (read to memory).

        Parameters
        ----------
        remove : `bool`
            Remove the memmap file after read values.
        """
        if not self.memmap:
            return

        self._contained = delete_array_memmap(self._contained, read=True,
                                              remove=remove)
        self._memmap = False

    def flush(self):
        """Write changes to disk if memmapping."""
        if self.memmap:
            self._contained.flush()

    def reset_data(self, data=None, dtype=None):
        """Set new data.

        Parameters
        ----------
        data : `~numpy.ndarray` or `None` (optional)
            Data array to be owned. If None, the container will be empty.
        dtype : `string` or `numpy.dtype` (optional)
            Imposed data type.
        """
        if data is None:
            if self.memmap:
                # Need to delete memmap
                mm = self._contained
                delete_array_memmap(mm, read=False, remove=False)
                self._contained = None
            else:
                # don't need to delete memmap
                self._contained = None

        # Not None data
        else:
            adata = data
            if isinstance(data, MemMapArray):
                adata = data._contained
            if not hasattr(adata, 'dtype'):
                adata = np.array(adata)
            if not adata.dtype.isnative:
                adata = adata.byteswap().newbyteorder()
                dtype = adata.dtype
            if self.memmap:
                name = self.filename
                mm = self._contained
                self._contained = create_array_memmap(name, adata, dtype)
                delete_array_memmap(mm, read=False, remove=True)
            else:
                self._contained = np.array(adata, dtype=dtype)

    def __getitem__(self, item):
        if self.empty:
            raise EmptyDataError('Empty data contaier')

        # This cannot create a new MemMapArray to don't break a[x][y] = z
        result = self._contained[item]
        return result

    def __setitem__(self, item, value):
        if self.empty:
            raise EmptyDataError('Empty data container')
        self._contained[item] = value

    def __getattribute__(self, item):
        if item in redirects and not self.empty:
            attr = getattr(self._contained, item)
            if callable(attr):
                attr = to_memmap_attr(attr)
            elif isinstance(attr, np.ndarray):
                attr = MemMapArray(attr, memmap=False)
            return attr
        if item in redirects and self.empty:
            raise EmptyDataError('Empty data container')

        return object.__getattribute__(self, item)

    def __repr__(self):
        return 'MemMapArray:\n' + repr(self._contained) + \
               f'\nfile: {self.filename}'

    def __array__(self, dtype=None):
        if self.empty:
            return np.array(None)
        # Ignore memmapping
        return np.array(self._contained, dtype=dtype)

    def __del__(self):
        """Safe destruct the MemMapArray."""
        try:
            os.remove(self._filename)
        except (FileNotFoundError, TypeError):
            pass

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
