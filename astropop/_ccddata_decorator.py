# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Custom CCDData class to support memmapping."""

import os
import shutil
import numpy as np
from astropy.nddata.ccddata import CCDData as AsCCDData
from tempfile import mkdtemp, mkstemp

from .py_utils import mkdir_p


__all__ = ['CCDData', 'memmapped_arrays']


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


def _cache_folder(obj, cache_folder):
    cache_folder = cache_folder or obj._cache
    cache_folder = cache_folder or mkdtemp(prefix='astropop')
    obj._cache = cache_folder
    mkdir_p(cache_folder)
    return cache_folder


def _filename(obj, filename, cache_folder):
    filename = filename or obj._basename
    if filename is None:
        f = mkstemp(prefix='ccddata', suffix='.npy',
                    dir=cache_folder)
        filename = f[1]
        os.remove(filename)
        obj._basename = filename
    else:
        filename = os.path.join(cache_folder,
                                os.path.basename(filename))
    return filename


def _check_memmap(obj, filename=None, cache_folder=None):
    """Check if all mappable objects are mapped"""

    # If is not memmapping already, create memmap
    cache_folder = _cache_folder(obj, cache_folder)
    filename = _filename(obj, filename, cache_folder)

    if obj.mapping:
        for name in obj._mapped_props:
            pname = '_{}'.format(name)
            dt = obj._fixed_dtype.get(name, None)
            _prop = getattr(obj, pname)
            if isinstance(_prop, np.memmap):
                continue
            if _prop is None:
                continue
            fname = filename + '.{}'.format(name)
            if dt is not None:
                dt = _prop.dtype
            data = _prop.astype(dt)
            setattr(obj, pname, create_array_memmap(fname, data))
    else:
        obj._disable_memmap()


def _enable_memmap(obj, filename=None, cache_folder=None):
    """Enable array file memmapping."""
    obj.mapping = True
    obj._check_memmap(filename, cache_folder)



def _disable_memmap(obj):
    """Disable array file memmapping (load to memory)."""
    obj.mapping = False
    for name in obj._mapped_props:
        pname = '_{}'.format(name)
        if isinstance(getattr(obj, pname), np.memmap):
            setattr(obj, pname, delete_array_memmap(getattr(obj, pname)))


def _init(obj, *args, **kwargs):
    obj._cache = kwargs.pop('cache_folder', None)
    obj._basename = kwargs.pop('basename', None)
    memmap = kwargs.pop('memmap', False)
    super(obj.__class__, obj).__init__(*args, **kwargs)

    if memmap:
        obj.enable_memmap()


def memmapped_arrays(cls, names, fixed_dtype={}):
    """Decorator to easely enable and disable memmapping for arrays."""
    cls.enable_memmap = _enable_memmap
    cls.disable_memmap = _disable_memmap
    cls._check_memmap = _check_memmap
    cls.__init__ = _init
    cls._mapped_props = names
    cls.memmaping = False
    cls._fixed_dtype = fixed_dtype

    # for each property, create getter, setter and deleter
    for name in names:
        _name = '_{}'.format(name)
        f_dtype = cls._fixed_dtype.get(name, None)
        setattr(cls, _name, None)

        def setter(obj, value):
            # Simply assign if not mapping
            if not obj.memmaping:
                setattr(obj, _name, value)
            elif value is None:
                delete_array_memmap(getattr(obj, _name))
                setattr(obj, _name, value)
            # Mapping
            else:
                existing = getattr(obj, _name)
                fname = getattr(obj, '_basename')+name
                dtype = f_dtype or value.dtype
                if existing is None:
                    nmap = create_array_memmap(fname, value.astype(dtype))
                    setattr(obj, _name, nmap)
                elif existing.shape != value.shape or existing.dtype != dtype:
                    delete_array_memmap(getattr(obj, _name))
                    nmap = create_array_memmap(fname, value.astype(dtype))
                    setattr(obj, _name, nmap)
                else:
                    getattr(obj, _name)[:] = value[:]
            obj._check_memmap()

        def getter(obj):
            return getattr(obj, _name)

        def deleter(obj):
            pp = getattr(obj, _name)
            if isinstance(pp, np.memmap):
                n = pp.filename
                dn = os.path.dirname(n)
                del pp
                os.remove(n)

                if len(os.listdir(dn)) == 0:
                    shutil.rmtree(dn)

        # Set the property to the class
        setattr(cls.__class__, name,
                property(fget=getter, fset=setter, fdel=deleter))

    return cls


CCDData = memmapped_arrays(AsCCDData, ['data', 'mask'],
                           fixed_dtype={'mask': bool})
