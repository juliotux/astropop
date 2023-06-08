"""Custom frame class to support memmapping.

We reimplemented and renamed astropy's frame to better handle disk memmap and
an easier unit/uncertainty workflow
"""

# Important:

import os
import numpy as np
import tempfile
from enum import Flag
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from ..flags import mask_from_flags
from ._memmap import create_array_memmap, delete_array_memmap, \
                     reset_memmap_array
from .compat import _merge_and_clean_header
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
        automatically. If using any quantity with masks, you must set the mask
        in a separate parameter.
    unit : `~astropy.units.Unit` or string (optional)
        The data unit. Must be `~astropy.units.Unit` compliant.
    dtype : string or `~numpy.dtype` (optional)
        Mandatory dtype of the data.
    uncertainty : array_like or `~astropy.nddata.Uncertanty` or `None` \
                    (optional)
        Uncertainty of the data.
    mask : array_like or `None` (optional)
        Frame mask. All the masked pixels will be flagged as
        `PixelMaskFlags.MASKED` and `PixelMaskFlags.UNSPECIFIED`.
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
    origin_filename : string, `~pathlib.Path` or `None` (optional)
        Original file name of the data. If set, it will be stored in the
        `FrameData` metadata.
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

    def __init__(self, data, unit=None, dtype=None, uncertainty=None,
                 mask=None, flags=None, wcs=None, meta=None, header=None,
                 cache_folder=None, cache_filename=None, origin_filename=None,
                 use_memmap_backend=False):

        # setup names
        self.cache_folder = cache_folder
        self.cache_filename = cache_filename
        self._origin = origin_filename

        # Ensure data is not None
        if data is None:
            raise ValueError('Data cannot be None.')
        if len(np.shape(data)) != 2:
            raise ValueError('Data must be 2D.')

        # raise errors if incompatible shapes
        data, uncertainty, mask, flags = shape_consistency(data, uncertainty,
                                                           mask, flags)
        # raise errors if incompatible units
        unit = extract_units(data, unit)
        self.unit = unit

        # setup flags and mask
        if flags is None:
            flags = np.zeros_like(data, dtype=np.uint8)

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
        self._header_update(header, meta, wcs)

    def _header_update(self, header, meta=None, wcs=None):
        # merge header and meta. meta with higher priority
        meta, wcs, history, comment = _merge_and_clean_header(meta, header,
                                                              wcs)
        if len(history) > 0:
            self.history = history
        if len(comment) > 0:
            self.comment = comment
        if wcs is not None:
            self._wcs = wcs
        self._meta = meta

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

    def astype(self, dtype):
        """Return a copy of the current FrameData with new dtype in data."""
        return self.copy(dtype)

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
        self._header_update(None, value)

    @property
    def header(self):
        """Get the header (metadata) of the frame."""
        return self._meta

    @property
    def history(self):
        """Get the FrameData stored history."""
        return self._history

    @history.setter
    def history(self, value):
        if not np.isscalar(value):
            self._history = self._history + list(value)
        else:
            self._history.append(value)

    @property
    def comment(self):
        """Get the FrameData stored comments."""
        return self._comments

    @comment.setter
    def comment(self, value):
        if not np.isscalar(value):
            self._comments = self._comments + list(value)
        else:
            self._comments.append(value)

    @property
    def data(self):
        """Get the main data container."""
        return self._data

    @data.setter
    def data(self, value):
        if hasattr(value, 'unit'):
            dunit = extract_units(value, None)
            self.unit = dunit
        self._data = reset_memmap_array(self._data, value)

    @property
    def uncertainty(self):
        """Get the uncertainty frame container."""
        return self._unct

    @uncertainty.setter
    def uncertainty(self, value):
        self._unct = delete_array_memmap(self._unct, read=False, remove=True)
        if value is None:
            return

        # Put is valid containers
        if np.isscalar(value):
            value = float(value)
        else:
            value = np.asarray(value, dtype=self.dtype)

        # units and shape must be consistent
        value = uncertainty_unit_consistency(self.unit, value)
        _, value, _, _ = shape_consistency(self.data, uncertainty=value)

        # ensure to memmap if already memmapping
        if self._memmapping:
            self._unct = create_array_memmap(value, self.cache_folder,
                                             self.cache_filename)
        else:
            self._unct = value

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
        if np.isscalar(self._unct):
            if self._unct is None and return_none:
                return None
            elif self._unct is None:
                return np.zeros_like(self.data)
            return self._unct
        return np.array(self._unct, dtype=self.dtype)

    @property
    def flags(self):
        """Get the flags frame container."""
        return self._flags

    @flags.setter
    def flags(self, value):
        self._flags = delete_array_memmap(self._flags, read=False, remove=True)
        if value is None:
            return

        # Put is valid containers
        if np.isscalar(value):
            value = int(value)
        else:
            value = np.asarray(value, dtype=np.uint8)
        _, _, _, flags = shape_consistency(self.data, flags=value)

        # ensure to memmap if already memmapping
        if self._memmapping:
            self._flags = create_array_memmap(value, self.cache_folder,
                                              self.cache_filename)
        else:
            self._flags = value

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

    @property
    def mask(self):
        """Mask all flagged pixels. True for all masked/removed pixels."""
        return self.mask_flags(PixelMaskFlags.MASKED)

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

    def enable_memmap(self, filename=None, cache_folder=None):
        """Enable array file memmapping.

        Parameters
        ----------
        filename : str, optional
            Custom memmap file name.
        cache_folder : str, optional
            Custom folder to cache data.
        """
        # early return for already memmapped
        if self._memmapping:
            return

        # get the default files if names are not provided
        if filename is None:
            filename = self.cache_filename
        if cache_folder is None:
            cache_folder = self.cache_folder

        # delete the old memmap files if they are memmaps
        self.disable_memmap()

        # setup the new filenames
        cache_file = setup_filename(self, cache_folder, filename)

        # create the memmap files
        self._data = create_array_memmap(filename=cache_file + '.data',
                                         data=self._data)
        self._flags = create_array_memmap(filename=cache_file + '.flags',
                                          data=self._flags)
        self._unct = create_array_memmap(filename=cache_file + '.unct',
                                         data=self._unct)
        self._memmapping = True

    def disable_memmap(self):
        """Disable frame file memmapping (load to memory)."""
        self._data = delete_array_memmap(self._data, read=True, remove=True)
        self._flags = delete_array_memmap(self._flags, read=True, remove=True)
        self._unct = delete_array_memmap(self._unct, read=True, remove=True)
        self._memmapping = False

    def copy(self, dtype=None):
        """Copy the current FrameData to a new instance.

        Parameters
        ----------
        - dtype: `~numpy.dtype` (optional)
            Data type for the copied FrameData. If `None`, the data type will
            be the same as the original Framedata.
            Default: `None`
        """
        nf = FrameData.__new__()
        # copy metadata
        nf._history = self._history.__copy__()
        nf._comments = self._comments.copy()
        nf._meta = self._meta.copy()
        nf._wcs = self._wcs.deepcopy()
        nf._unit = self._unit

        # file naming
        cache_fname = self.cache_filename
        if cache_fname is not None:
            cache_fname = cache_fname + '_copy'
        nf._origin = self._origin
        nf.cache_folder = self.cache_folder
        nf.cache_filename = cache_fname

        # copy data
        nf._data = delete_array_memmap(self._data, read=True, remove=False)
        nf._flags = delete_array_memmap(self._flags, read=True, remove=False)
        if self._unct is not None:
            nf._unct = delete_array_memmap(self._unct, read=True, remove=False)

        # copy memmapping
        if self._memmapping:
            nf.enable_memmap()

        return nf

    def __copy__(self):
        """Copy the current instance to a new one."""
        return self.copy()

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
