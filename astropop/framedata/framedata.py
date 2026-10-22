"""Custom frame class to support memmapping.

We reimplemented and renamed astropy's frame to better handle disk memmap and
an easier unit/uncertainty workflow
"""

from __future__ import annotations

import typing as _t
import numpy.typing as _npt
from .._typing import ImageLike as _ImageLike
from .._typing import (NumberArray as _NumberArray, PathLike as _PathLike,
                       UnitLike as _UnitLike, HeaderLike as _HeaderLike,
                       Index as _Index)
from astropy.nddata import CCDData as _CCDData

# Important:

import os
import copy as cp
import numpy as np
from enum import Flag
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from ..config import AstropopConfig as conf, _optional
from ..flags import mask_from_flags
from .cache_manager import TempDir
from ._memmap import create_array_memmap, delete_array_memmap, \
                     reset_memmap_array
from ._compat import _to_hdu, _to_ccddata, _write_fits, \
                     _normalize_and_strip_dict, extract_header_wcs
from .._unit_property import unit_property


__all__ = ['FrameData', 'PixelMaskFlags']


def shape_consistency(
    data: _ImageLike | None=None,
    uncertainty: float | int | np.number | _ImageLike | None=None,
    mask: _ImageLike | None=None,
    flags: _ImageLike | None=None,
) -> tuple[_ImageLike | None, _ImageLike | None, _ImageLike | None, _ImageLike | None]:
    """Check shape consistency across `data`, `uncertaitny` and `mask`."""
    dshape = np.shape(data)

    def _check(
        arr: float | int | np.number | _ImageLike | None,
        name: str,
        replicate: bool=False,
    ) -> _ImageLike | None:
        """Validate an optional array against data, expanding a scalar when allowed."""
        if arr is None:
            return
        if arr is not None and data is None:
            raise ValueError(f'{name} set for an empty data.')
        ashape = np.shape(arr)

        # if replicate, create an array full of value
        if ashape == () and replicate:
            arr = np.full(dshape, fill_value=arr)
            ashape = arr.shape

        if ashape != dshape:
            raise ValueError(f'{name} shape {ashape} don\'t match'
                             f' Data shape {dshape}.')
        return arr

    uncertainty = _check(uncertainty, 'Uncertainty', replicate=True)
    mask = _check(mask, 'Mask', replicate=False)
    flags = _check(flags, 'Flags', replicate=False)

    return data, uncertainty, mask, flags


def extract_units(
    data: _ImageLike | u.Quantity,
    unit: _UnitLike,
) -> u.UnitBase | None:
    """Extract and compare units if they are consistent."""
    if hasattr(data, 'unit'):
        dunit = u.Unit(data.unit)
    else:
        dunit = None
    if unit is not None:
        unit = u.Unit(unit)
    else:
        unit = None

    if dunit is not None and unit is not None:
        if dunit is not unit:
            raise ValueError(f"Unit {unit} cannot be set for a data with"
                             f" unit {dunit}")

    # if dunit is None, assign unit to it.
    if dunit is None:
        dunit = unit

    return dunit


def uncertainty_unit_consistency(
    unit: _UnitLike,
    uncertainty: float | int | np.number | _ImageLike | u.Quantity,
) -> _npt.NDArray:
    """Check if an uncertainty can be set based on its unit.

    Parameters
    ----------
    unit
        Target uncertainty unit. Mainly, this is the FrameData unit.
    uncertainty
        Uncertainty object. If it has units, it will be converted to the
        target.

    Returns
    -------
    uncertainty
        Uncertainty converted to the target unit.
    """
    # Assume a plain array uncertainty if no unit.
    if not hasattr(uncertainty, 'unit'):
        return np.array(uncertainty)

    u_unit = uncertainty.unit
    if u_unit == unit:
        return np.array(uncertainty)

    return u.Quantity(uncertainty, u_unit).to(unit).value


def setup_filename(
    frame: FrameData,
    cache_folder: str | TempDir | None=None,
    filename: _PathLike | None=None,
) -> str:
    """Handle filename and cache folder to a frame."""
    if not isinstance(frame, FrameData):
        raise ValueError('Only FrameData accepted.')

    cache_folder_ccd = frame.cache
    filename_ccd = frame.cache_filename

    # explicit set must be over defult
    filename = filename or filename_ccd
    if filename is None:
        # generate a random name
        filename = os.urandom(8).hex()
    # use only basename
    filename = os.path.basename(filename)

    # explicit set must be over defult
    # folders are automatically created
    cache_folder = cache_folder or cache_folder_ccd
    if cache_folder is None:
        cache = TempDir('framedata_'+filename)
    elif isinstance(cache_folder, str):
        cache = TempDir(cache_folder)
    elif isinstance(cache_folder, TempDir):
        cache = cache_folder
    else:
        raise ValueError('cache_folder must be a string'
                         ' or a TempDir instance.')

    # Setup the FrameData values.
    frame.cache = cache
    frame.cache_filename = filename

    return os.path.join(str(cache_folder), filename)


class PixelMaskFlags(Flag):
    """Flags for pixel masking.

    Examples
    --------
    >>> from astropop.framedata import PixelMaskFlags as F

    >>> # Record saturation and request exclusion as separate bits.
    >>> bits = F.SATURATED | F.MASKED
    >>> print(bits.value, F.MASKED in bits)
    18 True
    """
    # Type of Masking
    INTERPOLATED = 1 << 0  # pixel interpolated from neighbors
    MASKED = 1 << 1  # pixel value removed
    REMOVED = 1 << 1  # same of masked. Both are equal
    # Cause of problem
    DEAD = 1 << 2  # dead pixel
    BAD = 1 << 3  # bad pixel
    SATURATED = 1 << 4  # saturated pixel, above a threshold level
    COSMIC_RAY = 1 << 5  # cosmic ray
    OUT_OF_BOUNDS = 1 << 6  # registered image. Pixel is out of the bounds
    UNSPECIFIED = 1 << 7  # not specified


# Store this flags as uint8
PixelMaskFlags.dtype = np.uint8


@unit_property
class FrameData:
    """Two-dimensional image with units, optional arrays, and disk caching.

    Uncertainties are stored as standard-deviation arrays. Pixel flags encode
    both masking and pixel status. Metadata and WCS accompany the image.

    Parameters
    ----------
    data
        The main data values. If `~astropy.units.Quantity`, unit will be set
        automatically. If using any quantity with masks, you must set the mask
        in a separate parameter.
    unit
        The data unit. Must be `~astropy.units.Unit` compliant.
    dtype
        Integer or floating-point dtype of the data. Default: float64.
        Integer uncertainties use the same dtype and truncate fractions.
    uncertainty
        Standard deviation in the data unit. Scalars are expanded to the
        image shape; arrays must match it. Default: None.
    mask
        Frame mask. All the masked pixels will be flagged as
        `PixelMaskFlags.MASKED` and `PixelMaskFlags.UNSPECIFIED`.
    flags
        Pixel flags matching the image shape. See `PixelMaskFlags` for values.
        Stored as uint8; defaults to zeros unless flag storage is disabled.
    wcs : `~astropy.wcs.WCS` or `None` (optional)
        World Coordinate System of the image. If meta or header keys already
        contain WCS information, an error will be raised.
    meta, header : dict or astropy.io.fits.Header, optional
        Metadata (header) of the frame. Only one accepted. If both are passed,
        error will be raised.
    cache_folder
        Directory for cached arrays. A temporary directory under the Astropy
        cache is created by default. Use a dedicated directory: cache cleanup
        removes the directory and its contents.
    cache_filename
        Base file name to store the cached `~astropop.framedata.FrameData`.
    origin_filename
        Original file name of the data, available as ``origin_filename``.
    use_memmap_backend
        Enable memmapping in the constructor using the configured cache folder.

    Notes
    -----
    - Data and uncertainty share the same physical unit.
    - Built-in arithmetic operators are not supported. Use
      `~astropop.image.imarith.imarith` for arithmetic and propagation.
    - Optional uncertainty and flag storage follow the process-wide
      `~astropop.config.AstropopConfig` settings. Set them before loading or
      creating frames to avoid allocating these arrays. Changing the settings
      afterward hides existing arrays without freeing their memory.

    Examples
    --------
    >>> from astropop.framedata import FrameData
    >>> frame = FrameData([[10., 20.]], unit='adu', uncertainty=2.)

    >>> # A scalar standard uncertainty is expanded to the image shape.
    >>> print(frame.shape, frame.uncertainty.tolist())
    (1, 2) [[2.0, 2.0]]

    >>> frame.mask_pixels((0, 1))
    >>> print(frame.mask.tolist(), frame.data.tolist())
    [[False, True]] [[10.0, 20.0]]
    """

    # The decorator installs this descriptor at runtime; expose both access
    # contracts to static tools without replacing the runtime implementation.
    if _t.TYPE_CHECKING:
        @property
        def unit(self) -> u.UnitBase:
            """Return the physical unit, using dimensionless units when none was stored."""
            ...

        @unit.setter
        def unit(self, value: _UnitLike) -> None:
            """Normalize and replace the physical unit without rescaling pixel values."""
            ...

    _memmapping = False
    _unit = None
    _data = None
    _flags = None
    _unct = None
    _wcs = None
    _meta = None
    _origin = None
    _history = None
    _comments = None
    cache = None
    cache_filename = None
    memmap_objects = None

    def __init__(
        self,
        data: _ImageLike | u.Quantity,
        unit: _UnitLike=None,
        dtype: _npt.DTypeLike | None=None,
        uncertainty: float | int | np.number | _ImageLike | u.Quantity | None=None,
        mask: _ImageLike | None=None,
        flags: _ImageLike | None=None,
        use_memmap_backend: bool=False,
        cache_folder: str | TempDir | None=None,
        cache_filename: _PathLike | None=None,
        origin_filename: _PathLike | None=None,
        **kwargs: _t.Any,
    ) -> None:
        """Initialize image storage, metadata and enabled optional arrays.

        The class documentation describes constructor parameters and configuration.
        Disabled optional arrays are discarded before validation or allocation.
        Memory mapping is enabled only when explicitly requested.
        """
        # setup names
        setup_filename(self, cache_folder, cache_filename)
        self._origin = origin_filename

        # Ensure data is not None
        if len(np.shape(data)) != 2:
            raise ValueError('Data must be 2D array.')
        if dtype is not None:
            dtype = np.dtype(dtype)
            if dtype.kind not in 'iuf':
                raise ValueError('Data dtype must be integer or float.')
        else:
            dtype = np.dtype('f8')
        if dtype.kind in 'iu' and not np.all(np.isfinite(data)):
            raise ValueError('Integer data require finite input pixels.')

        # Disabled arrays must not be validated, converted, or allocated.
        if conf.FRAMEDATA_DISABLE_UNCERTAINTY:
            uncertainty = None
        if conf.FRAMEDATA_DISABLE_FLAGS:
            flags = mask = None

        # raise errors if incompatible shapes
        data, uncertainty, mask, flags = shape_consistency(data, uncertainty,
                                                           mask, flags)
        # raise errors if incompatible units
        unit = extract_units(data, unit)
        self.unit = unit

        # setup flags and mask
        if flags is None and not conf.FRAMEDATA_DISABLE_FLAGS:
            flags = np.zeros_like(data, dtype=PixelMaskFlags.dtype)

        # set data to the variables
        self._data = np.asarray(data, dtype=dtype)
        self.flags = flags
        self.uncertainty = uncertainty
        # create flag for masked pixels
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            self.add_flags(PixelMaskFlags.MASKED | PixelMaskFlags.UNSPECIFIED,
                           mask)

        # Check for memmapping.
        self._memmapping = False
        if use_memmap_backend:
            self.enable_memmap()

        # avoiding security problems
        self._history = []
        self._comments = []
        self._meta = fits.Header()
        if 'meta' in kwargs and 'header' in kwargs:
            raise ValueError('Only one of meta or header can be set.')
        if 'meta' not in kwargs:
            kwargs['meta'] = kwargs.pop('header', None)
        self._meta_update(kwargs.pop('meta', None),
                          kwargs.pop('wcs', None))

    def _meta_update(self, meta: _HeaderLike | None, wcs: WCS | None=None) -> None:
        """Normalize FITS metadata, append commentary and separate the WCS."""
        # simplify things by enforcing meta type
        if not isinstance(meta, (dict, fits.Header, type(None))):
            raise TypeError('meta must be a dict, Header or None.')

        # force fits compliant header
        try:
            if meta is not None:
                hdr = fits.Header(meta)
            else:
                hdr = fits.Header()
        except Exception as e:
            raise ValueError('meta or header must be compilant with FITS '
                             'standard. Got error when tried to convert to '
                             f'fits.Header: {e}')

        header, history, comment = _normalize_and_strip_dict(hdr)
        if len(history) > 0:
            self.history = history
        if len(comment) > 0:
            self.comment = comment

        header, _wcs = extract_header_wcs(header)
        if _wcs is not None and wcs is not None:
            raise ValueError('wcs and meta/wcs cannot be set at the same '
                             'time.')
        self._wcs = _wcs or wcs
        self._meta = header

    @property
    def origin_filename(self) -> _PathLike | None:
        """Get the original filename of the data."""
        return self._origin

    @property
    def shape(self) -> tuple[int, int]:
        """Get the data shape following numpy. ``FrameData.data.shape``."""
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        """Get the dta type of the data. ``FrameData.data.dtype``."""
        return self._data.dtype

    def astype(self, dtype: _npt.DTypeLike) -> FrameData:
        """Return a copy of the current FrameData with new dtype in data.

        Examples
        --------
        >>> from astropop.framedata import FrameData
        >>> frame = FrameData([[10., 20.]], uncertainty=0.5)
        >>> converted = frame.astype('float32')
        >>> print(converted.dtype.name, converted.uncertainty.dtype.name)
        float32 float32
        """
        return self.copy(dtype)

    @property
    def size(self) -> int:
        """Get the size of the data. ``FrameData.data.size``."""
        return self._data.size

    @property
    def wcs(self) -> WCS | None:
        """Get the World Coordinate System."""
        return self._wcs

    @wcs.setter
    def wcs(self, value: WCS | None) -> None:
        """Assign the WCS or clear it with None; reject other objects."""
        if isinstance(value, WCS) or value is None:
            self._wcs = value
        else:
            raise TypeError('wcs setter value must be a WCS instance.')

    @property
    def meta(self) -> fits.Header:
        """Get the metadata (header) of the frame."""
        return self._meta

    @meta.setter
    def meta(self, value: _HeaderLike | None) -> None:
        """Replace normalized metadata, append its commentary and extract its WCS."""
        self._meta_update(value)

    @property
    def header(self) -> fits.Header:
        """Get the header (metadata) of the frame."""
        return self._meta

    @property
    def history(self) -> list[str]:
        """Get the FrameData stored history."""
        return self._history

    @history.setter
    def history(self, value: str | _t.Iterable[str]) -> None:
        """Append one or more history entries; ignore None."""
        if value is None:
            return
        if not np.isscalar(value):
            self._history = self._history + list(value)
        else:
            self._history.append(value)

    @property
    def comment(self) -> list[str]:
        """Get the FrameData stored comments."""
        return self._comments

    @comment.setter
    def comment(self, value: str | _t.Iterable[str]) -> None:
        """Append one or more comment entries; ignore None."""
        if value is None:
            return
        if not np.isscalar(value):
            self._comments = self._comments + list(value)
        else:
            self._comments.append(value)

    @property
    def data(self) -> _npt.NDArray:
        """Get the main data container."""
        return self._data

    @data.setter
    def data(self, value: _ImageLike) -> None:
        """Replace pixel storage and adopt an attached unit without propagating errors."""
        if hasattr(value, 'unit'):
            dunit = extract_units(value, None)
            self.unit = dunit
        self._data = reset_memmap_array(self._data, value)

    def get_masked_data(self, fill_value: float | int=np.nan) -> _npt.NDArray:
        """Copy pixel values, replacing MASKED pixels with a fill value.

        Parameters
        ----------
        fill_value
            Value assigned to masked pixels in the copy. Default: numpy.nan.

        Returns
        -------
        data
            Independent data array. The stored pixels are unchanged. Integer data
            with an enabled mask are promoted to floating point for a nonfinite
            fill; finite fills retain the data dtype. With flags absent or disabled,
            return an unchanged copy of the data.
        """
        d = self.data.copy()
        mask = self.mask
        if mask is not None:
            if self.dtype.kind in 'iu' and not np.isfinite(fill_value):
                d = d.astype(float)
            d[mask] = fill_value
        return d

    @property
    @_optional('FRAMEDATA_DISABLE_UNCERTAINTY')
    def uncertainty(self) -> _npt.NDArray | None:
        """Get the uncertainty array, or None when absent or disabled."""
        return self._unct

    @uncertainty.setter
    def uncertainty(self, value: float | int | np.number | _ImageLike | u.Quantity | None) -> None:
        """Replace standard uncertainties, checking units and data shape.

        A scalar is expanded to the image shape. None or disabled uncertainty storage
        removes the existing array and its backing file. Integer storage requires
        finite values.
        """
        if value is None or conf.FRAMEDATA_DISABLE_UNCERTAINTY:
            self._unct = delete_array_memmap(self._unct, read=False, remove=True)
            return

        if self.dtype.kind in 'iu':
            value = uncertainty_unit_consistency(self.unit, value)
            if not np.all(np.isfinite(value)):
                raise ValueError('Integer uncertainties require finite values.')
            value = np.asarray(value, dtype=self.dtype)
        elif np.isscalar(value):
            value = float(value)
        else:
            value = np.asarray(value, dtype=self.dtype)

        # units and shape must be consistent
        value = uncertainty_unit_consistency(self.unit, value)
        _, value, _, _ = shape_consistency(self.data, uncertainty=value)

        # Copy a possible view before closing/replacing the original mapping.
        self._unct = reset_memmap_array(self._unct, value)
        self._update_memmaps()

    def get_uncertainty(self, return_none: bool=True) -> _npt.NDArray | None:
        """Return a copy of the standard uncertainties or an explicit fallback.

        Parameters
        ----------
        return_none
            Return None for missing or disabled uncertainty. If False, allocate
            a zero-filled array instead. Default: True.

        Returns
        -------
        uncertainty
            Independent array with the data shape and dtype, or None. Returning
            zeros is an explicit allocation even when storage is disabled.

        Examples
        --------
        >>> from astropop.framedata import FrameData
        >>> frame = FrameData([[10., 20.]])
        >>> print(frame.get_uncertainty())
        None

        >>> # Request zeros for calculations without allocating stored errors.
        >>> print(frame.get_uncertainty(False).tolist())
        [[0.0, 0.0]]
        >>> print(frame.uncertainty)
        None
        """
        uncertainty = self.uncertainty
        if uncertainty is None and return_none:
            return None
        elif uncertainty is None:
            return np.zeros_like(self.data)
        return np.array(uncertainty, dtype=self.dtype)

    @property
    @_optional('FRAMEDATA_DISABLE_FLAGS')
    def flags(self) -> _npt.NDArray[np.uint8] | None:
        """Get the flags array, or None when absent or disabled."""
        return self._flags

    @flags.setter
    def flags(self, value: _ImageLike | None) -> None:
        """Replace shape-matched flag bits using the pixel-flag dtype.

        None or disabled flag storage removes the existing array and its backing
        file. Scalar flags and values carrying physical units are rejected.
        """
        if value is None or conf.FRAMEDATA_DISABLE_FLAGS:
            self._flags = delete_array_memmap(self._flags, read=False, remove=True)
            return

        # Put is valid containers
        if np.isscalar(value):
            raise ValueError('Flags cannot be scalar.')
        elif hasattr(value, 'unit'):
            raise ValueError('Flags cannot have units.')
        else:
            value = np.asarray(value, dtype=PixelMaskFlags.dtype)
        _, _, _, flags = shape_consistency(self.data, flags=value)

        self._flags = reset_memmap_array(self._flags, value,
                                          dtype=PixelMaskFlags.dtype)
        self._update_memmaps()

    def add_flags(self, flag: PixelMaskFlags, where: _Index) -> None:
        """Add a given flag to the pixels in the given positions.

        Parameters
        ----------
        flag
            Flag to be added.
        where
            Positions where the flag will be added.

        Examples
        --------
        >>> from astropop.framedata import FrameData, PixelMaskFlags as F
        >>> frame = FrameData([[10., 100.]])

        >>> # Mark the reason and exclusion policy together at bright pixels.
        >>> frame.add_flags(F.SATURATED | F.MASKED, where=frame.data > 50)
        >>> print(frame.flags.tolist(), frame.mask.tolist())
        [[0, 18]] [[False, True]]
        """
        if not isinstance(flag, PixelMaskFlags):
            raise TypeError('Flag must be a PixelMaskFlags instance.')
        if conf.FRAMEDATA_DISABLE_FLAGS:
            self.flags = None
            return
        if self._flags is None:
            self.flags = np.zeros(self.shape, dtype=PixelMaskFlags.dtype)
        self._flags[where] |= flag.value

    @property
    def mask(self) -> _npt.NDArray[np.bool_] | None:
        """Return masked pixels, or None when flags are absent or disabled."""
        return self.mask_flags(PixelMaskFlags.MASKED)

    def mask_flags(
        self,
        flags: PixelMaskFlags | _t.Sequence[PixelMaskFlags],
    ) -> _npt.NDArray[np.bool_] | None:
        """Get a mask pixels with an specific flag.

        Parameters
        ----------
        flags

        Returns
        -------
        mask
            True for masked pixels; None when flags are absent or disabled.
        """
        pixel_flags = self.flags
        if pixel_flags is None:
            return None
        return mask_from_flags(pixel_flags, flags,
                               allowed_flags_class=PixelMaskFlags)

    def mask_pixels(self, pixels: _Index) -> None:
        """Mask pixels.

        Parameters
        ----------
        pixels
            Pixels to be masked. Can be a tuple of (y, x) positions or a
            boolean array where True means masked. Uses the same standard as
            ``array[pixels]`` indexing.
        """
        self.add_flags(PixelMaskFlags.MASKED, pixels)

    def enable_memmap(
        self,
        cache_folder: str | TempDir | None=None,
        filename: _PathLike | None=None,
    ) -> None:
        """Enable array file memmapping.

        Parameters
        ----------
        filename
            Custom memmap file name.
        cache_folder
            Custom folder to cache data.
        """
        # early return for already memmapped
        if self._memmapping:
            return

        # re-setup the filenames
        setup_filename(self, cache_folder, filename)

        # delete the old memmap files if they are memmaps
        self.disable_memmap()

        # create the memmap files
        self._memmapping = True
        self._update_memmaps()

    def disable_memmap(self) -> None:
        """Disable frame file memmapping (load to memory)."""
        self._data = delete_array_memmap(self._data, read=True, remove=True)
        self._flags = delete_array_memmap(self._flags, read=True, remove=True)
        self._unct = delete_array_memmap(self._unct, read=True, remove=True)
        self._memmapping = False

    def _update_memmaps(self) -> None:
        """Update the memmap files."""

        if self._memmapping:
            for attr, suffix in (('_data', 'data'), ('_flags', 'flags'),
                                 ('_unct', 'unct')):
                value = getattr(self, attr)
                if value is not None and not isinstance(value, np.memmap):
                    filename = self.cache.create_file(
                        self.cache_filename + f'.{suffix}.npy')
                    setattr(self, attr, create_array_memmap(str(filename), value))

    def copy(self, dtype: _npt.DTypeLike | None=None) -> FrameData:
        """Copy data, enabled optional arrays, metadata and WCS independently.

        Parameters
        ----------
        dtype
            Convert the copied data and present uncertainties to this dtype.
            If None, keep their existing dtypes. Default: None.

        Returns
        -------
        frame
            Independent frame with a separate cache directory. Memory-mapped
            sources produce a copy with its own mapped files. Optional-array
            configuration is evaluated when copying, so hidden arrays are omitted.
            Origin filename, history and comments are preserved.

        Examples
        --------
        >>> from astropop.framedata import FrameData
        >>> original = FrameData([[10.]])
        >>> duplicate = original.copy()

        >>> # Editing copied pixels leaves the original frame unchanged.
        >>> duplicate.data[0, 0] = 99
        >>> print(original.data.tolist(), duplicate.data.tolist())
        [[10.0]] [[99.0]]
        """
        nf = object.__new__(FrameData)
        # copy metadata
        nf._history = cp.deepcopy(self._history)
        nf._comments = cp.deepcopy(self._comments)
        nf._meta = cp.deepcopy(self._meta)
        nf._wcs = cp.deepcopy(self._wcs)
        nf._unit = self._unit

        # file naming
        cache_fname = self.cache_filename
        if cache_fname is not None:
            cache_fname = cache_fname + '_copy'
        nf._origin = self._origin
        if self.cache is None:
            setup_filename(nf, filename=cache_fname)
        else:
            cache_dir = self.cache.dirname + '_copy'
            if os.path.exists(self.cache.full_path + '_copy'):
                cache_dir += '_' + os.urandom(8).hex()
            nf.cache = TempDir(cache_dir, parent=self.cache.parent)
            nf.cache_filename = cache_fname

        # copy data
        nf._data = delete_array_memmap(self._data, read=True, remove=False)
        if dtype is not None:
            nf._data = nf._data.astype(dtype)
        nf._flags = delete_array_memmap(self.flags, read=True, remove=False)
        uncertainty = self.uncertainty
        if uncertainty is not None:
            nf._unct = delete_array_memmap(uncertainty, read=True, remove=False)
            if dtype is not None:
                nf._unct = nf._unct.astype(dtype)

        # copy memmapping
        if self._memmapping:
            nf.enable_memmap()

        return nf

    def __copy__(self) -> FrameData:
        """Copy the current instance to a new one."""
        return self.copy()

    def __deepcopy__(self, memo: dict[int, object]) -> FrameData:
        """Copy arrays and metadata without sharing memory maps."""
        result = self.copy()
        memo[id(self)] = result
        return result

    def __del__(self) -> None:
        """Release memory mappings before deleting the managed cache directory."""
        if self.cache is not None:
            # Release mappings before deleting their files on Windows. There
            # is no need to copy the arrays when destroying the frame.
            if self._memmapping:
                for array in (self._data, self._flags, self._unct):
                    if isinstance(array, np.memmap) and array._mmap is not None:
                        array._mmap.close()
            self.cache.delete()

    def median(self, **kwargs: _t.Any) -> _NumberArray | u.Quantity:
        """Compute and return the median of the data."""
        med = np.median(self._data, **kwargs)
        if self._unit is None:
            return med
        return med * self.unit

    def mean(self, **kwargs: _t.Any) -> _NumberArray | u.Quantity:
        """Compute and return the mean of the data."""
        if self._unit is None:
            return self._data.mean(**kwargs)
        return self._data.mean(**kwargs) * self.unit

    def std(self, **kwargs: _t.Any) -> _NumberArray | u.Quantity:
        """Compute and return the std dev of the data."""
        if self._unit is None:
            return self._data.std(**kwargs)
        return self._data.std(**kwargs) * self.unit

    def min(self) -> _NumberArray | u.Quantity:
        """Compute minimum value of the data."""
        if self._unit is None:
            return self._data.min()
        return self._data.min() * self.unit

    def max(self) -> _NumberArray | u.Quantity:
        """Compute minimum value of the data."""
        if self._unit is None:
            return self._data.max()
        return self._data.max() * self.unit

    def statistics(self) -> dict[str, _NumberArray | u.Quantity]:
        """Compute general statistics ofthe image."""
        return {'min': self.min(),
                'max': self.max(),
                'mean': self.mean(),
                'median': self.median(),
                'std': self.std()}

    def to_hdu(
        self,
        wcs_relax: bool | int=True,
        no_fits_standard_units: bool=True,
        **kwargs: _t.Any,
    ) -> fits.HDUList:
        """Generate an HDUList from this FrameData.

        Parameters
        ----------
        wcs_relax
            Allow non-standard WCS keys.
            Default: `True`
        no_fits_standard_units
            Allow Astropy-compatible unit strings instead of requiring
            FITS-standard unit formatting.
            Default: `True`
        **kwargs
            hdu_uncertainty: string, optional
                Standard-deviation extension name (default: ``UNCERT``).
                Set to None to omit this extension.
            hdu_mask: string, optional
                Boolean mask extension name (default: ``MASK``).
                Set to None to omit this extension. Individual pixel flag
                bits are not serialized.
            unit_key: string, optional
                Header key for physical unit.


        Returns
        -------
        result
            HDU list containing data, metadata, WCS, and present optional arrays.

        Examples
        --------
        >>> from astropop.framedata import FrameData
        >>> frame = FrameData([[10.]], unit='adu', uncertainty=2.)
        >>> with frame.to_hdu() as hdus:
        ...     print(hdus[0].data.tolist(), hdus['UNCERT'].data.tolist())
        [[10.0]] [[2.0]]
        """
        return _to_hdu(self, wcs_relax=wcs_relax,
                       no_fits_standard_units=no_fits_standard_units,
                       **kwargs)

    def to_ccddata(self) -> _CCDData:
        """Convert this FrameData to CCDData.

        Returns
        -------
        result
            CCDData with data, unit, metadata, WCS, standard-deviation
            uncertainty, and Boolean mask. Missing optional arrays remain
            None. Individual flag bits and the separate history and comment
            lists are not transferred.
        """
        return _to_ccddata(self)

    def write(
        self,
        filename: _PathLike,
        overwrite: bool=False,
        no_fits_standard_units: bool=True,
        **kwargs: _t.Any,
    ) -> None:
        """Write frame to a fits file.

        Parameters
        ----------
        filename
            Name of the file to write.
        overwrite
            If True, overwrite the file if it exists.
        wcs_relax: `bool`, optional.
            Allow non-standard WCS keys.
            Default: `True`
        no_fits_standard_units
            Allow Astropy-compatible unit strings instead of requiring
            FITS-standard unit formatting.
            Default: `True`
        **kwargs
            hdu_uncertainty: string, optional
                Standard-deviation extension name (default: ``UNCERT``).
                Set to None to omit this extension.
            hdu_mask: string, optional
                Boolean mask extension name (default: ``MASK``).
                Set to None to omit this extension. Individual pixel flag
                bits are not serialized.
            unit_key: string, optional
                Header key for physical unit.
        """
        _write_fits(self, filename, overwrite,
                    no_fits_standard_units=no_fits_standard_units,
                    **kwargs)
