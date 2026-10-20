# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools to handle memmaped arrays."""

from __future__ import annotations

import os as _os
import numpy.typing as _npt

import os
import numpy as np


__all__ = ['create_array_memmap', 'delete_array_memmap', 'reset_memmap_array']


def create_array_memmap(
    filename: str | _os.PathLike[str] | None,
    data: _npt.ArrayLike | None,
    dtype: _npt.DTypeLike | None=None,
) -> np.memmap | _npt.ArrayLike | None:
    """Create a memory map to an array data.

    Parameters
    ----------
    filename
        Name of memmap file to be created.
    data
        Data to be stored in the memmap.
    dtype
        `~numpy.dtype` compilant data type. If `None`, `data.dtype` will be
        used.

    Returns
    -------
    memmap
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


def delete_array_memmap(
    memmap: _npt.ArrayLike | None,
    read: bool=True,
    remove: bool=False,
) -> _npt.ArrayLike | None:
    """Delete a memmap and read the data to a np.ndarray.

    Parameters
    ----------
    memmap
        MemMap array to be deleted.
    read
        Read the data to memory before delete.
    remove
        Close the mapping and delete the memmap file. References to the
        mapping, including views, must not be used afterwards.

    Returns
    -------
    data
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
        # Deleting a local reference leaves the caller's mapping open.
        # Windows requires the mapping to be closed before unlinking its file.
        if memmap._mmap is not None:
            memmap._mmap.close()
        if os.path.exists(name):
            os.remove(name)
    return data


def reset_memmap_array(
    array: _npt.ArrayLike | None,
    data: _npt.ArrayLike | None,
    dtype: _npt.DTypeLike | None=None,
) -> _npt.ArrayLike | None:
    """Reset a memmap array to a new data.

    Parameters
    ----------
    array
        MemMap array to be reseted.
    data
        Data to be stored in the memmap.

    Returns
    -------
    memmap
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
