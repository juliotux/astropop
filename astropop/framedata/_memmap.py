# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools to handle memmaped arrays."""

import os
import numpy as np


__all__ = ['create_array_memmap', 'delete_array_memmap', 'reset_memmap_array']


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
    # scalars should not be memmaped
    if np.isscalar(data) or data is None:
        return data

    if filename is None:
        raise ValueError('None filename')

    data = np.array(data)
    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = data.dtype
    # check native byteorder. Needed for some FITS applications
    if not dtype.isnative:
        dtype = dtype.newbyteorder('=')

    shape = data.shape
    memmap = np.memmap(str(filename), mode='w+', dtype=dtype, shape=shape)
    memmap[:] = data[:]
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
    if memmap is None or np.isscalar(memmap):
        return memmap

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


def reset_memmap_array(array, data, dtype=None):
    """Reset a memmap array to a new data.

    Parameters
    ----------
    array : array_like
        MemMap array to be reseted.
    data : array_like
        Data to be stored in the memmap.

    Returns
    -------
    memmap : `~numpy.memmap`
        Memmap object of cached data.
    """
    # scalar not memmaped
    if np.isscalar(data) or data is None:
        if isinstance(array, np.memmap):
            # Need to delete memmap
            delete_array_memmap(array, read=False, remove=True)
        return data

    data = np.array(data)
    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = data.dtype
    # check native byteorder. Needed for some FITS applications
    if not dtype.isnative:
        dtype = dtype.newbyteorder('=')

    # return according memmap
    if isinstance(array, np.memmap):
        name = array.filename
        array = delete_array_memmap(array, read=False, remove=True)
        return create_array_memmap(name, data, dtype)
    return np.array(data, dtype=dtype)
