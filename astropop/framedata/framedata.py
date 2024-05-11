"""Custom frame class to support memmapping.

We reimplemented and renamed astropy's frame to better handle disk memmap and
an easier unit/uncertainty workflow
"""

# Important:

import os
import copy as cp
import numpy as np
from enum import Flag
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from ..flags import mask_from_flags
from .cache_manager import TempDir
from ._memmap import create_array_memmap, delete_array_memmap, \
                     reset_memmap_array
from ._compat import _to_hdu, _to_ccddata, _write_fits, \
                     _normalize_and_strip_dict, extract_header_wcs
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


# Store this flags as uint8
PixelMaskFlags.dtype = np.uint8


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
        World Coordinate System of the image. If meta or header keys already
        contain WCS informations, an error will be raised.
    meta or header: `dict` or `astropy.fits.Header` (optional)
        Metadata (header) of the frame. Only one accepted. If both are passed,
        error will be raised.
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
    cache = None
    cache_filename = None
    memmap_objects = None

    def __init__(self, data, unit=None, dtype=None, uncertainty=None,
                 mask=None, flags=None, use_memmap_backend=False,
                 cache_folder=None, cache_filename=None, origin_filename=None,
                 **kwargs):
        # setup names
        setup_filename(self, cache_folder, cache_filename)
        self._origin = origin_filename

        # Ensure data is not None
        if len(np.shape(data)) != 2:
            raise ValueError('Data must be 2D array.')
        if dtype is not None:
            dtype = np.dtype(dtype)
            if dtype.kind != 'f':
                raise ValueError('Data dtype must be float.')
        else:
            dtype = np.dtype('f8')

        # raise errors if incompatible shapes
        data, uncertainty, mask, flags = shape_consistency(data, uncertainty,
                                                           mask, flags)
        # raise errors if incompatible units
        unit = extract_units(data, unit)
        self.unit = unit

        # setup flags and mask
        if flags is None:
            flags = np.zeros_like(data, dtype=PixelMaskFlags.dtype)

        # set data to the variables
        self._data = np.asarray(data, dtype=dtype)
        self.flags = np.asarray(flags, dtype=PixelMaskFlags.dtype)
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

    def _meta_update(self, meta, wcs=None):
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
        self._meta_update(value)

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
        if value is None:
            return
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
        if value is None:
            return
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

    def get_masked_data(self, fill_value=np.nan):
        """Return a copy of the data with masked pixels as `fill_value`."""
        d = self.data.copy()
        d[self.mask] = fill_value
        return d

    @property
    def uncertainty(self):
        """Get the uncertainty frame container."""
        return self._unct

    @uncertainty.setter
    def uncertainty(self, value):
        if value is None:
            self._unct = None
            # do not delete earlier to avoid security issues if other parts of
            # the code raises error
            delete_array_memmap(self._unct, read=False, remove=True)
            return

        # Put is valid containers
        if np.isscalar(value):
            value = float(value)
        else:
            value = np.asarray(value, dtype=self.dtype)

        # units and shape must be consistent
        value = uncertainty_unit_consistency(self.unit, value)
        _, value, _, _ = shape_consistency(self.data, uncertainty=value)

        # ensure that the memmap is removed
        self._unct = delete_array_memmap(self._unct, read=False, remove=True)
        self._unct = value
        self._update_memmaps()

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
        if self._unct is None and return_none:
            return None
        elif self._unct is None:
            return np.zeros_like(self.data)
        return np.array(self._unct, dtype=self.dtype)

    @property
    def flags(self):
        """Get the flags frame container."""
        return self._flags

    @flags.setter
    def flags(self, value):
        if value is None:
            self._flags = None
            # do not delete earlier to avoid security issues if other parts of
            # the code raises error
            delete_array_memmap(self._flags, read=False, remove=True)
            return

        # Put is valid containers
        if np.isscalar(value):
            raise ValueError('Flags cannot be scalar.')
        elif hasattr(value, 'unit'):
            raise ValueError('Flags cannot have units.')
        else:
            value = np.asarray(value, dtype=PixelMaskFlags.dtype)
        _, _, _, flags = shape_consistency(self.data, flags=value)

        self._flags = value
        self._update_memmaps()

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
        """Get a mask pixels with an specific flag.

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

    def mask_pixels(self, pixels):
        """Mask pixels.

        Parameters
        ----------
        pixels: `~numpy.ndarray` or tuple
            Pixels to be masked. Can be a tuple of (y, x) positions or a
            boolean array where True means masked. Uses the same standard as
            `~numpy.ndarray`[pixels] access.
        """
        self._flags[pixels] |= PixelMaskFlags.MASKED.value

    def enable_memmap(self, cache_folder=None, filename=None):
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

        # re-setup the filenames
        setup_filename(self, cache_folder, filename)

        # delete the old memmap files if they are memmaps
        self.disable_memmap()

        # create the memmap files
        self._memmapping = True
        self._update_memmaps()

    def disable_memmap(self):
        """Disable frame file memmapping (load to memory)."""
        self._data = delete_array_memmap(self._data, read=True, remove=True)
        self._flags = delete_array_memmap(self._flags, read=True, remove=True)
        self._unct = delete_array_memmap(self._unct, read=True, remove=True)
        self._memmapping = False

    def _update_memmaps(self):
        """Update the memmap files."""

        if self._memmapping:
            # get the default files if names are not provided
            dataf = self.cache.create_file(self.cache_filename + '.data.npy')
            flagf = self.cache.create_file(self.cache_filename + '.flags.npy')
            unctf = self.cache.create_file(self.cache_filename + '.unct.npy')
            if not isinstance(self._data, np.memmap):
                self._data = create_array_memmap(str(dataf), self._data)
            if not isinstance(self._flags, np.memmap) and \
               self._flags is not None:
                self._flags = create_array_memmap(str(flagf), self._flags)
            if not isinstance(self._unct, np.memmap) and \
               self._unct is not None:
                self._unct = create_array_memmap(str(unctf), self._unct)

    def copy(self, dtype=None):
        """Copy the current FrameData to a new instance.

        Parameters
        ----------
        - dtype: `~numpy.dtype` (optional)
            Data type for the copied FrameData. If `None`, the data type will
            be the same as the original Framedata.
            Default: `None`
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
        nf.cache = TempDir(self.cache.dirname + '_copy',
                           parent=self.cache.parent)
        nf.cache_filename = cache_fname

        # copy data
        nf._data = delete_array_memmap(self._data, read=True, remove=False)
        if dtype is not None:
            nf._data = nf._data.astype(dtype)
        nf._flags = delete_array_memmap(self._flags, read=True, remove=False)
        if self._unct is not None:
            nf._unct = delete_array_memmap(self._unct, read=True, remove=False)
            if dtype is not None:
                nf._unct = nf._unct.astype(dtype)

        # copy memmapping
        if self._memmapping:
            nf.enable_memmap()

        return nf

    def __copy__(self):
        """Copy the current instance to a new one."""
        return self.copy()

    def __del__(self):
        if self.cache is not None:
            self.cache.delete()

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
                Extension name to store the uncertainty in 2D image format.
            hdu_flags: string, optional
                Extension name to store the pixel list flags in table format.
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

    def to_ccddata(self):
        """Convert actual FrameData to CCDData.

        Returns
        -------
        `~astropy.nddata.CCDData` :
            CCDData instance with actual FrameData informations.
        """
        return _to_ccddata(self)

    def write(self, filename, overwrite=False, no_fits_standard_units=True,
              **kwargs):
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
                Extension name to store the uncertainty in 2D image format.
            hdu_flags: string, optional
                Extension name to store the pixel list flags in table format.
            unit_key: string, optional
                Header key for physical unit.
        """
        _write_fits(self, filename, overwrite,
                    no_fits_standard_units=no_fits_standard_units,
                    **kwargs)
