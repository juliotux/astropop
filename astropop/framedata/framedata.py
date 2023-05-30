# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Custom frame class to support memmapping."""

# Important: We reimplemented and renamed astropy's frame to better
# handle disk memmap and an easier unit/uncertainty workflow

import os
import numpy as np
import copy as cp
import tempfile
from enum import Flag
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from ..py_utils import check_iterable
from ..flags import mask_from_flags
from .memmap import MemMapArray
from .compat import _to_ccddata, _to_hdu, _merge_and_clean_header, _write_fits
from .._unit_property import unit_property


__all__ = ['FrameData', 'PixelMaskFlags']


def shape_consistency(data=None, uncertainty=None, mask=None, flags=None):
    """Check shape consistency across `data`, `uncertaitny` and `mask`."""
    dshape = np.shape(data)

    def _check(arr, name, replicate=False):
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


def extract_units(data, unit):
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


def uncertainty_unit_consistency(unit, uncertainty):
    """Check if an uncertainty can be set based on its unit.

    Parameters
    ----------
    unit: string or `~astropy.units.Unit`
        Target uncertainty unit. Mainly, this is the FrameData unit.
    uncertainty: array_like
        Uncertainty object. If it has units, it will be converted to the
        target.

    Returns
    -------
    uncertainty: `numpy.ndarray`
        Uncertainty converted to the target unit.
    """
    # Assume a plain array uncertainty if no unit.
    if not hasattr(uncertainty, 'unit'):
        return np.array(uncertainty)

    u_unit = uncertainty.unit
    if u_unit == unit:
        return np.array(uncertainty)

    return u.Quantity(uncertainty, u_unit).to(unit).value


def setup_filename(frame, cache_folder=None, filename=None):
    """Handle filename and cache folder to a frame."""
    if not isinstance(frame, FrameData):
        raise ValueError('Only FrameData accepted.')

    cache_folder_ccd = frame.cache_folder
    filename_ccd = frame.cache_filename

    # explicit set must be over defult
    filename = filename or filename_ccd
    if filename is None:
        filename = tempfile.NamedTemporaryFile(suffix='.npy').name

    if cache_folder is None and \
       os.path.dirname(filename) not in ('', tempfile.gettempdir()):
        # we need filename dir for cache_folder
        cache_folder = os.path.dirname(filename)

    filename = os.path.basename(filename)

    # explicit set must be over defult
    cache_folder = cache_folder or cache_folder_ccd
    if cache_folder is None:
        cache_folder = tempfile.TemporaryDirectory(prefix='astropop_').name

    os.makedirs(cache_folder, exist_ok=True)

    frame.cache_folder = cache_folder
    frame.cache_filename = filename

    return os.path.join(cache_folder, filename)


class PixelMaskFlags(Flag):
    """Flags for pixel masking."""
    DEAD = 1 << 0  # dead pixel
    BAD = 1 << 1  # bad pixel
    SATURATED = 1 << 2  # saturated pixel, above a threshold level
    INTERPOLATED = 1 << 3  # pixel interpolated from neighbors
    COSMIC_RAY = 1 << 4  # cosmic ray
    OUT_OF_BOUNDS = 1 << 5  # registered image. Pixel is out of the bounds
    MASKED = 1 << 6  # not specified


@unit_property
class FrameData:
    """Data container for image frame to handle memmapping data from disk.

    The main difference from Astropy's `~astropy.nddata.CCDData` is the
    memmapping itself. However it handles uncertainties in a totally different
    way. It stores only StdDev uncertainty arrays. It also stores the unit.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The main data values. If `~astropy.units.Quantity`, unit will be set
        automatically.
    unit : `~astropy.units.Unit` or string (optional)
        The data unit. Must be `~astropy.units.Unit` compliant.
    dtype : string or `~numpy.dtype` (optional)
        Mandatory dtype of the data.
    uncertainty : array_like or `~astropy.nddata.Uncertanty` or `None` \
                    (optional)
        Uncertainty of the data.
    u_dtype : string or `~numpy.dtype` (optional)
        Mandatory dtype of uncertainty.
    mask : array_like or `None` (optional)
        Frame mask. All the masked pixels will be flagged as
        `PixelMaskFlags.MASKED`.
    flags : array_like or `None` (optional)
        Pixel flags for the frame. See `~astropop.FrameData.PixelMaskFlags`.
        for values.
    wcs : `dict`, `~astropy.fits.Header` or `~astropy.wcs.WCS` (optional)
        World Coordinate System of the image.
    meta or header: `dict` or `astropy.fits.Header` (optional)
        Metadata (header) of the frame. If both set, they will be merged.
        `header` priority.
    cache_folder : string, `~pathlib.Path` or `None` (optional)
        Place to store the cached `FrameData`
    cache_filename : string, `~pathlib.Path` or `None` (optional)
        Base file name to store the cached `FrameData`.
    use_memmap_backend : `bool` (optional)
        True if enable memmap in constructor.

    Notes
    -----
    - The physical unit is assumed to be the same for data and uncertainty.
      So, we droped the support for data with data with different uncertainty
      unit, like `~astropy.nddata.ccddata.CCDData` does.
    - As this is intended to be a safe container for data, it do not handle
      builtin math operations. For math operations using FrameData, check
      `~astropop.ccd_processing.imarith` module.
    """

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

    def __init__(self, data, unit=None, dtype=None,
                 uncertainty=None, u_dtype=None, mask=None, flags=None,
                 wcs=None, meta=None, header=None,
                 cache_folder=None, cache_filename=None,
                 use_memmap_backend=False, origin_filename=None):
        # TODO: implement pixel list
        self.cache_folder = cache_folder
        self.cache_filename = cache_filename
        self._origin = origin_filename

        # Setup MemMapArray instances
        cache_file = setup_filename(self, self.cache_folder,
                                    self.cache_filename)
        self._update_cache_files(cache_file)

        # Check for memmapping.
        self._memmapping = False
        if use_memmap_backend:
            self.enable_memmap()

        # Masking handle
        if hasattr(data, 'mask'):
            dmask = data.mask
            if mask is not None:
                mask = np.logical_or(dmask, mask)
            else:
                mask = dmask

        # raise errors if incompatible shapes
        data, uncertainty, mask, flags = shape_consistency(data, uncertainty,
                                                           mask, flags)
        if flags is None:
            flags = np.zeros_like(data, dtype=np.uint8)

        # raise errors if incompatible units
        unit = extract_units(data, unit)
        self.unit = unit
        self._data.reset_data(data, dtype)
        self._unct.reset_data(uncertainty, u_dtype)
        self._flags.reset_data(flags, np.uint8)

        # create flag for masked pixels
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            self.add_flags(PixelMaskFlags.MASKED, mask)

        # avoiding security problems
        self._history = []
        self._comments = []
        self._meta = fits.Header()
        self._header_update(header, meta, wcs)

    def _header_update(self, header, meta=None, wcs=None):
        # merge header and meta. meta with higher priority
        meta, wcs, history, comment = _merge_and_clean_header(meta, header,
                                                              wcs)
        if len(history) > 0:
            self.history = history
        if len(comment) > 0:
            self.comments = comment
        if wcs is not None:
            self._wcs = wcs
        self._meta = meta

    def _update_cache_files(self, cache_file):
        # TODO: if cache folder is changing, remove it
        if self._data is not None:
            self._data.disable_memmap(remove=True)
            nd = self._data._contained
        else:
            nd = None
        if self._unct is not None:
            self._unct.disable_memmap(remove=True)
            nu = self._unct._contained
        else:
            nu = None
        if self._flags is not None:
            self._flags.disable_memmap(remove=True)
            nm = self._flags._contained
        else:
            nm = None

        self._data = MemMapArray(nd, filename=cache_file + '.data')
        self._unct = MemMapArray(nu, filename=cache_file + '.unct')
        self._flags = MemMapArray(nm, filename=cache_file + '.flags')

    @property
    def history(self):
        """Get the FrameData stored history."""
        return self._history

    @history.setter
    def history(self, value):
        if check_iterable(value):
            self._history = self._history + list(value)
        else:
            self._history.append(value)

    @property
    def comment(self):
        """Get the FrameData stored comments."""
        return self._comments

    @comment.setter
    def comment(self, value):
        if check_iterable(value):
            self._comments = self._comments + list(value)
        else:
            self._comments.append(value)

    @property
    def origin_filename(self):
        """Get the original filename of the data."""
        return self._origin

    @property
    def shape(self):
        """Get the data shape following numpy. `FrameData.data.shape`."""
        return self._data.shape

    @property
    def dtype(self):
        """Get the dta type of the data. `FrameData.data.dtype`."""
        return self._data.dtype

    @property
    def size(self):
        """Get the size of the data. `FrameData.data.size`."""
        return self._data.size

    @property
    def wcs(self):
        """Get the World Coordinate System."""
        return self._wcs

    @wcs.setter
    def wcs(self, value):
        if isinstance(value, WCS) or value is None:
            self._wcs = value
        else:
            raise TypeError('wcs setter value must be a WCS instance.')

    @property
    def meta(self):
        """Get the metadata (header) of the frame."""
        return self._meta

    @meta.setter
    def meta(self, value):
        self._header_update(value)

    @property
    def header(self):
        """Get the header (metadata) of the frame."""
        return self._meta

    @property
    def data(self):
        """Get the main data container."""
        return self._data

    @data.setter
    def data(self, value):
        if hasattr(value, 'unit'):
            dunit = extract_units(value, None)
            self.unit = dunit
        self._data.reset_data(value)

    @property
    def uncertainty(self):
        """Get the uncertainty frame container."""
        return self._unct

    @uncertainty.setter
    def uncertainty(self, value):
        if value is None:
            self._unct.reset_data(value)
        else:
            _, value, _, _ = shape_consistency(self.data, value, None)
            value = uncertainty_unit_consistency(self.unit, value)
            self._unct.reset_data(value)

    def get_uncertainty(self, return_none=True):
        """Get the uncertainty frame in a safer way.

        In some cases, like interfacing with QFloat, the uncertainty cannot be
        None or an empty container. So, in cases like this, is prefered to get
        a whole matrix containing zeroes. This method is responsible for this
        special returns. For non-empty containers, a copy in `~numpy.ndarray`
        container will be returned.

        Parameters
        ----------
        - return_none: bool (optional)
          If True, an empty uncertainty frame will return only None. Else, a
          matrix filled with zeroes will be returned.
          Default: True
        """
        if self._unct.empty and return_none:
            return None
        if self._unct.empty:
            return np.zeros_like(self._data, dtype=self.dtype)
        return np.array(self._unct, dtype=self.dtype)

    @property
    def mask(self):
        """Mask all flagged pixels. True for masked pixels."""
        return np.bool_(self._flags)

    def mask_flags(self, flags):
        """Mask pixels with an specific flag.

        Parameters
        ----------
        flags: list of `PixelMaskFlags` or `PixelMaskFlags`

        Returns
        -------
        mask: `~numpy.ndarray`
            Masked pixels. True for masked pixels.
        """
        return mask_from_flags(self._flags, flags,
                               allowed_flags_class=PixelMaskFlags)

    @property
    def flags(self):
        """Get the flags frame container."""
        return self._flags

    @flags.setter
    def flags(self, value):
        if np.array(value).dtype.kind != 'u':
            raise TypeError('Flags must be an unsigned integer, not '
                            f'{np.array(value).dtype}.')
        if value is None:
            self._flags.reset_data(value, dtype=np.uint8)
        else:
            _, _, _, flags = shape_consistency(self.data, flags=value)
            self._flags.reset_data(value, dtype=np.uint8)

    def add_flags(self, flag, where):
        """Add a given flag to the pixels in the given positions.

        Parameters
        ----------
        flag : `PixelMaskFlags`
            Flag to be added.
        where : `~numpy.ndarray`
            Positions where the flag will be added.
        """
        if not isinstance(flag, PixelMaskFlags):
            raise TypeError('Flag must be a PixelMaskFlags instance.')
        self._flags[where] |= flag.value

    def astype(self, dtype):
        """Return a copy of the current FrameData with new dtype in data."""
        return self.copy(dtype)

    def copy(self, dtype=None):
        """Copy the current FrameData to a new instance.

        Parameters
        ----------
        - dtype: `~numpy.dtype` (optional)
            Data type for the copied FrameData. If `None`, the data type will
            be the same as the original Framedata.
            Default: `None`
        """
        data = np.array(self._data) if not self._data.empty else None
        flags = np.array(self._flags) if not self._flags.empty else None
        unct = np.array(self._unct) if not self._unct.empty else None
        unit = self._unit
        wcs = cp.copy(self._wcs)
        meta = fits.Header(self._meta, copy=True)
        hist = cp.copy(self._history)
        comm = cp.copy(self._comments)
        fname = self._origin
        cache_folder = self.cache_folder
        cache_fname = self.cache_filename

        if dtype is not None:
            data = data.astype(dtype) if data is not None else data
            unct = unct.astype(dtype) if unct is not None else unct

        if cache_fname is not None:
            cache_fname = cache_fname + '_copy'

        nframe = FrameData(data, unit=unit, flags=flags, uncertainty=unct,
                           meta=meta, cache_folder=cache_folder,
                           cache_filename=cache_fname, origin_filename=fname)
        nframe.history = hist
        nframe.comments = comm
        nframe.wcs = wcs
        return nframe

    def enable_memmap(self, filename=None, cache_folder=None):
        """Enable array file memmapping.

        Parameters
        ----------
        filename : str, optional
            Custom memmap file name.
        cache_folder : str, optional
            Custom folder to cache data.
        """
        if filename is None and cache_folder is None:
            self._data.enable_memmap()
            self._unct.enable_memmap()
            self._flags.enable_memmap()
        else:
            cache_file = setup_filename(self, cache_folder, filename)
            self._update_cache_files(cache_file)
            self._data.enable_memmap(cache_file + '.data')
            self._flags.enable_memmap(cache_file + '.flags')
            self._unct.enable_memmap(cache_file + '.unct')
        self._memmapping = True

    def disable_memmap(self):
        """Disable frame file memmapping (load to memory)."""
        self._data.disable_memmap(remove=True)
        self._flags.disable_memmap(remove=True)
        self._unct.disable_memmap(remove=True)
        self._memmapping = False

    def to_hdu(self, wcs_relax=True, no_fits_standard_units=True, **kwargs):
        """Generate an HDUList from this FrameData.

        Parameters
        ----------
        wcs_relax: `bool`, optional.
            Allow non-standard WCS keys.
            Default: `True`
        no_fits_standard_units: `bool`, optional
            Skip FITS units standard for units. If this options is choose,
            the units will be printed in header as `~astropy.units.Unit`
            compatible string.
            Default: `True`
        **kwargs:
            hdu_uncertainty: string, optional
                Extension name to store the uncertainty.
            hdu_mask: string, optional
                Extension name to store the mask.
            unit_key: string, optional
                Header key for physical unit.


        Returns
        -------
        `~astropy.fits.HDUList` :
            HDU storing all FrameData informations.
        """
        return _to_hdu(self, wcs_relax=wcs_relax,
                       no_fits_standard_units=no_fits_standard_units,
                       **kwargs)

    def __del__(self):
        """Safe destruction of the container."""
        # ensure all files are removed when exit
        self.disable_memmap()
        # remove tmp folder if empty
        try:
            if len(os.listdir(self.cache_folder)) == 0:
                os.rmdir(self.cache_folder)
        except FileNotFoundError:
            pass

    def to_ccddata(self):
        """Convert actual FrameData to CCDData.

        Returns
        -------
        `~astropy.nddata.CCDData` :
            CCDData instance with actual FrameData informations.
        """
        return _to_ccddata(self)

    def write(self, filename, overwrite=False, **kwargs):
        """Write frame to a fits file.

        Parameters
        ----------
        filename: str
            Name of the file to write.
        overwrite: bool, optional
            If True, overwrite the file if it exists.
        wcs_relax: `bool`, optional.
            Allow non-standard WCS keys.
            Default: `True`
        no_fits_standard_units: `bool`, optional
            Skip FITS units standard for units. If this options is choose,
            the units will be printed in header as `~astropy.units.Unit`
            compatible string.
            Default: `True`
        **kwargs:
            hdu_uncertainty: string, optional
                Extension name to store the uncertainty.
            hdu_mask: string, optional
                Extension name to store the mask.
            unit_key: string, optional
                Header key for physical unit.
        """
        _write_fits(self, filename, overwrite, **kwargs)

    def __copy__(self):
        """Copy the current instance to a new one."""
        return self.copy()

    def median(self, **kwargs):
        """Compute and return the median of the data."""
        med = np.median(self._data, **kwargs)
        if self._unit is None:
            return med
        return med * self.unit

    def mean(self, **kwargs):
        """Compute and return the mean of the data."""
        if self._unit is None:
            return self._data.mean(**kwargs)
        return self._data.mean(**kwargs) * self.unit

    def std(self, **kwargs):
        """Compute and return the std dev of the data."""
        if self._unit is None:
            return self._data.std(**kwargs)
        return self._data.std(**kwargs) * self.unit

    def min(self):
        """Compute minimum value of the data."""
        if self._unit is None:
            return self._data.min()
        return self._data.min() * self.unit

    def max(self):
        """Compute minimum value of the data."""
        if self._unit is None:
            return self._data.max()
        return self._data.max() * self.unit

    def statistics(self):
        """Compute general statistics ofthe image."""
        return {'min': self.min(),
                'max': self.max(),
                'mean': self.mean(),
                'median': self.median(),
                'std': self.std()}
