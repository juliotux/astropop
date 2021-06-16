# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Custom frame class to support memmapping."""

# Important: We reimplemented and renamed astropy's frame to better
# handle disk memmap and an easier unit/uncertainty workflow

import os
import numpy as np
import copy as cp
from tempfile import mkdtemp, mkstemp
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import CCDData

from ..py_utils import mkdir_p
from .memmap import MemMapArray
from .compat import extract_header_wcs, _to_ccddata, _to_hdu, \
                    _extract_ccddata, _extract_fits, imhdus
from ._unit_property import unit_property


__all__ = ['FrameData', 'read_framedata', 'check_framedata']


def _get_shape(d):
    if hasattr(d, 'shape'):
        ds = d.shape
    else:
        ds = np.array(d).shape
    return ds


def read_framedata(obj, copy=False):
    """Read an object to a FrameData container.

    Parameters
    ----------
    - obj: any compatible, see notes
      Object that will be readed to the FrameData.
    - copy: bool (optional)
      If the object is already a FrameData, return a copy instead of the
      original one.
      Default: False

    Returns
    -------
    - frame: `FrameData`
      The readed FrameData object.

    Notes
    -----
    - If obj is a string or `~pathlib.Path`, it will be interpreted as a file.
      File types will be checked. Just FITS format supported now.
    - If obj is `~astropy.io.fits.HDUList`, `~astropy.io.fits.HDUList` or
      `~astropy.nddata.CCDData`, they will be properly translated to
      `FrameData`.
    - If numbers or `~astropop.math.physical.QFloat`,
      `~astropy.units.Quantity`, they will be translated to a `FrameData`
      without metadata.
    """
    if isinstance(obj, FrameData):
        if copy:
            obj = cp.deepcopy(obj)
    elif isinstance(obj, CCDData):
        obj = FrameData(**_extract_ccddata(obj))
    elif isinstance(obj, (str, bytes, os.PathLike)):
        obj = FrameData(**_extract_fits(obj))
    elif isinstance(obj, (fits.HDUList)+imhdus):
        obj = FrameData(**_extract_fits(obj))
    else:
        raise ValueError(f'Object {obj} is not compatible with FrameData.')

    # TODO: numbers, np.array, QFloat, Quantity

    return obj


check_framedata = read_framedata


def shape_consistency(data=None, uncertainty=None, mask=None):
    """Check shape consistency across `data`, `uncertaitny` and `mask`."""
    if data is None and uncertainty is not None:
        raise ValueError('Uncertainty set for an empty data.')
    if data is None and mask not in (None, False):
        raise ValueError('Mask set for an empty data.')

    dshape = _get_shape(data)

    if uncertainty is not None:
        ushape = _get_shape(uncertainty)

        if ushape == ():
            uncertainty = np.ones(dshape)*uncertainty
            ushape = uncertainty.shape

        if ushape != dshape:
            raise ValueError(f'Uncertainty shape {ushape} don\'t match'
                             f' Data shape {dshape}.')

    if mask is not None:
        mshape = _get_shape(mask)

        if mshape == ():
            mask = np.logical_or(np.zeros(dshape), mask)
            mshape = mask.shape

        if mask.shape != dshape:
            raise ValueError(f'Mask shape {mshape} don\'t match'
                             f' Data shape {dshape}.')

    return data, uncertainty, mask


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
    if hasattr(frame, 'cache_folder'):
        cache_folder_ccd = frame.cache_folder
    else:
        cache_folder_ccd = None

    if hasattr(frame, 'cache_filename'):
        filename_ccd = frame.cache_filename
    else:
        filename_ccd = None

    filename = filename_ccd or filename
    filename = filename or mkstemp(suffix='.npy')[1]
    filename = os.path.basename(filename)
    if cache_folder is None and os.path.dirname(filename) != '':
        cache_folder = os.path.dirname(filename)

    cache_folder = cache_folder_ccd or cache_folder
    cache_folder = cache_folder or mkdtemp(prefix='astropop')

    frame.cache_folder = cache_folder
    frame.cache_filename = filename

    mkdir_p(cache_folder)
    return os.path.join(cache_folder, filename)


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
        Frame mask.
    m_dtype : string or `~numpy.dtype` (optional)
        Mandatory dtype of mask. Default `bool`compilant
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

    # TODO: Complete reimplement the initialize
    # TODO: __copy__
    # TODO: write_fits

    _memmapping = False
    _unit = None
    _data = None
    _mask = None
    _unct = None
    _wcs = None
    _meta = {}
    _origin = None
    _history = []

    def __init__(self, data, unit=None, dtype=None,
                 uncertainty=None, u_dtype=None,
                 mask=None, m_dtype=bool,
                 wcs=None, meta=None, header=None,
                 cache_folder=None, cache_filename=None,
                 use_memmap_backend=False, origin_filename=None):

        self.cache_folder = cache_folder
        self.cache_filename = cache_filename
        self._origin = origin_filename

        # Setup MemMapArray instances
        cache_file = setup_filename(self, self.cache_folder,
                                    self.cache_filename)
        self._data = MemMapArray(None, filename=cache_file + '.data')
        self._unct = MemMapArray(None, filename=cache_file + '.unct')
        self._mask = MemMapArray(None, filename=cache_file + '.mask')

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
        if mask is None:  # Default do not mask anything
            mask = False

        # raise errors if incompatible shapes
        data, uncertainty, mask = shape_consistency(data, uncertainty, mask)
        # raise errors if incompatible units
        unit = extract_units(data, unit)
        self.unit = unit
        self._data.reset_data(data, dtype)
        self._unct.reset_data(uncertainty, u_dtype)
        # Masks can also be flags (uint8)
        self._mask.reset_data(mask, m_dtype)
        self._header_update(header, meta, wcs)

        self._history = []

    def _header_update(self, header, meta, wcs):
        if wcs is not None:
            self.wcs = wcs
        if meta is not None and header is not None:
            header = dict(header)
            meta = dict(meta)
            self._meta = meta
            self._meta.update(header)
        elif meta is not None:
            self._meta = dict(meta)
        elif header is not None:
            self._meta = dict(meta)
        else:
            self._meta = dict()

    @property
    def history(self):
        """Get the FrameData stored history."""
        return self._history

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
        if isinstance(value, WCS):
            self._wcs = value
        else:
            raise TypeError('wcs setter value must be a WCS instance.')

    @property
    def meta(self):
        """Get the metadata (header) of the frame."""
        return self._meta

    @meta.setter
    def meta(self, value):
        self.header = value

    @property
    def header(self):
        """Get the header (metadata) of the frame."""
        return self._meta

    @header.setter
    def header(self, value):
        value, wcs = extract_header_wcs(value)
        if wcs is not None:
            # If a WCS is found, overriding framedata WCS.
            self.wcs = wcs
        self._meta = dict(value)

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
        """Return the uncertainty.

        Note
        ----
        - If the uncertainty is an empty container, this property returns
          an `~numpy.zeros_like` array with the same shape of the data.
        """
        if self._unct.empty:
            # FIXME: this should be None, but conflicts with qfloat.
            return np.zeros_like(self._data)
        return self._unct

    @uncertainty.setter
    def uncertainty(self, value):
        if value is None:
            self._unct.reset_data(value)
        else:
            _, value, _ = shape_consistency(self.data, value, None)
            value = uncertainty_unit_consistency(self.unit, value)
            self._unct.reset_data(value)

    @property
    def mask(self):
        """Access mask data."""
        return self._mask

    @mask.setter
    def mask(self, value):
        _, _, value = shape_consistency(self.data, None, value)
        self._mask.reset_data(value)

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
            self._mask.enable_memmap()
        else:
            cache_file = setup_filename(self, cache_folder, filename)
            self._data.enable_memmap(cache_file + '.data')
            self._mask.enable_memmap(cache_file + '.mask')
            self._unct.enable_memmap(cache_file + '.unct')
        self._memmapping = True

    def disable_memmap(self):
        """Disable frame file memmapping (load to memory)."""
        self._data.disable_memmap(remove=True)
        self._mask.disable_memmap(remove=True)
        self._unct.disable_memmap(remove=True)
        self._memmapping = False

    def to_hdu(self, **kwargs):
        """Generate an HDUList from this FrameData.

        Parameters
        ----------
        hdu_uncertainty : string, optional
            Extension name to store the uncertainty. If None,
            no uncertainty will be stored.
        hdu_mask : string, optional
            Extension name to store the mask. If None, no mask
            will be saved.
        unit_key : string, optional
            Header key for physical unit.
        wcs_relax : `bool`, optional.
            Allow non-standard WCS keys.

        Returns
        -------
        `~astropy.fits.HDUList` :
            HDU storing all FrameData informations.
        """
        return _to_hdu(self, **kwargs)

    def to_ccddata(self):
        """Convert actual FrameData to CCDData.

        Returns
        -------
        `~astropy.nddata.CCDData` :
            CCDData instance with actual FrameData informations.
        """
        return _to_ccddata(self)
