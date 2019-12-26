# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Custom frame class to support memmapping."""

# Important: We reimplemented and renamed astropy's frame to better
# handle disk memmap and an easier unit/uncertainty workflow

import os
import numpy as np
from tempfile import mkdtemp, mkstemp
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.ccddata import _generate_wcs_and_update_header, CCDData

from ..py_utils import mkdir_p
from .memmap import MemMapArray


__all__ = ['FrameData', 'shape_consistency', 'unit_consistency',
           'setup_filename', 'extract_units']


def shape_consistency(data=None, uncertainty=None, mask=None):
    """Check shape consistency across ``data``, ``uncertaitny`` and ``mask``"""
    if data is None and uncertainty is not None:
        raise ValueError('Uncertainty set for an empty data.')
    if data is None and mask not in (None, False):
        raise ValueError('Mask set for an empty data.')

    if hasattr(data, 'shape'):
        dshape = data.shape
    else:
        dshape = np.array(data).shape

    if uncertainty is not None:
        if hasattr(uncertainty, 'shape'):
            ushape = uncertainty.shape
        else:
            ushape = np.array(uncertainty).shape

        if ushape == ():
            uncertainty = np.ones(dshape)*uncertainty
            ushape = uncertainty.shape

        if ushape != dshape:
            raise ValueError(f'Uncertainty shape {ushape} don\'t match'
                             f' Data shape {dshape}.')

    if mask is not None:
        if hasattr(mask, 'shape'):
            mshape = mask.shape
        else:
            mshape = np.array(mask).shape

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
        else:
            return dunit
    elif dunit is not None:
        return dunit
    elif unit is not None:
        return unit
    else:
        return None


def unit_consistency(data_unit=None, uncertainty_unit=None):
    '''Check physical unit consistency between data and uncertanty.'''
    if uncertainty_unit is None:
        # Uncertainty unit None is always compatible
        return
    elif data_unit is None:
        raise ValueError(f'Uncertainty with unit {uncertainty_unit} '
                         'incompatible with dimensionless data.')
    # Trick to make it run
    elif 1*u.Unit(data_unit) != 1*u.Unit(uncertainty_unit):
        raise ValueError(f'Units {data_unit} and {uncertainty_unit} '
                         'are incompatible')


def setup_filename(frame, cache_folder=None, filename=None):
    """Setup filename and cache folder to a frame"""
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


class FrameData:
    """Data conainer for image frame to handle memmapping data from disk.

    The main difference from Astropy's `~astropy.nddata.CCDData` is the
    memmapping itself. However it handles uncertainties in a totally different
    way. It stores only StdDev uncertainty arrays. It also stores the unit.

    Parameters:
    -----------
    - data : array_like or `astropy.units.Quantity`
        The main data values. If `Quantity`, unit will be set automatically.
    - unit : `astropy.units.Unit` or string (optional)
        The data unit. Must be `astropy.units.Unit` compilant.
    - dtype : string or `numpy.dtype` (optional)
        Mandatory dtype of the data.
    - uncertainty : array_like or `astropy.nddata.Uncertanty` or `None` \
                    (optional)
        Uncertainty of the data.
    - u_unit : `astropy.units.Unit` or string (optional)
        Unit of uncertainty of the data. If None, will be the same of the
        data.
    - u_dtype : string or `numpy.dtype` (optional)
        Mandatory dtype of uncertainty.
    - mask : array_like or `None` (optional)
        Frame mask.
    - m_dtype : string or `numpy.dtype` (optional)
        Mandatory dtype of mask. Default `bool`
    - wcs : dict, `astropy.fits.Header` or `astropy.wcs.WCS` (optional)
        World Coordinate System of the image.
    - meta or header: dict or `astropy.fits.Header` (optional)
        Metadata (header) of the frame. If both set, they will be merged.
        `header` priority.
    - cache_folder : string, `pathlib.Path` or `None` (optional)
        Place to store the cached `FrameData`
    - cache_filename : string, `pathlib.Path` or `None` (optional)
        Base file name to store the cached `FrameData`.
    - use_memmap_backend : `bool` (optional)
        True if enable memmap in constructor.
    """
    # TODO: Complete reimplement the initialize
    # TODO: __copy__
    _memmapping = False
    _data = None
    _mask = None
    _unct = None
    _wcs = None
    _meta = None
    _origin = None
    _history = None

    def __init__(self, data, unit=None, dtype=None,
                 uncertainty=None, u_unit=None, u_dtype=None,
                 mask=None, m_dtype=bool,
                 wcs=None, meta=None, header=None,
                 cache_folder=None, cache_filename=None,
                 use_memmap_backend=False, origin_filename=None):

        if isinstance(data, u.Quantity):
            raise TypeError('astropy Quantity not supported yet.')

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
        dunit = extract_units(data, unit)
        uunit = extract_units(data, u_unit)
        unit_consistency(dunit, uunit)

        self._data.reset_data(data, dunit, dtype)
        self._unct.reset_data(uncertainty, uunit, u_dtype)
        # Masks can also be flags (uint8)
        self._mask.reset_data(mask, None, m_dtype)

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
        return self._history

    @property
    def origin_filename(self):
        """Original filename of the data."""
        return self._origin

    @property
    def shape(self):
        """Data shape following numpy. Same as `FrameData.data.shape`"""
        return self._data.shape

    @property
    def dtype(self):
        """Data type of the data. Same as `FrameData.data.dtype`"""
        return self._data.dtype

    @property
    def size(self):
        """Size of the data. Same as `FrameData.data.size`"""
        return self._data.size

    @property
    def wcs(self):
        """Data World Coordinate System."""
        return self._wcs

    @wcs.setter
    def wcs(self, value):
        if isinstance(value, WCS):
            self._wcs = value
        else:
            raise TypeError('wcs setter value must be a WCS instance.')

    @property
    def meta(self):
        """Metadata (header) of the frame."""
        return self._meta

    @meta.setter
    def meta(self, value):
        self.header = value

    @property
    def header(self):
        """Header (metadata) of the frame."""
        return self._meta

    @header.setter
    def header(self, value):
        value, wcs = _generate_wcs_and_update_header(value)
        if wcs is not None:
            # If a WCS is found, overriding framedata WCS.
            self.wcs = wcs
        self._meta = dict(value)

    @property
    def data(self):
        """Main data array."""
        return self._data

    @data.setter
    def data(self, value):
        self._data.reset_data(value)

    @property
    def unit(self):
        """Physical unit of the data. Same as `FrameData.data.unit`."""
        return self._data.unit

    @property
    def uncertainty(self):
        """Data uncertainty."""
        if self._unct.empty:
            return 0.0*self._data.unit
        return self._unct

    @uncertainty.setter
    def uncertainty(self, value):
        if value is None:
            self._unct.reset_data(value)
        else:
            _, value, _ = shape_consistency(self.data, value, None)
            if hasattr(value, 'unit'):
                v_unit = value.unit
            else:
                # FIXME: Assume scalar value has the same unit of data?
                v_unit = self.data.unit
            unit_consistency(self.data.unit, v_unit)
            self._unct.reset_data(value, unit=v_unit)

    @property
    def mask(self):
        """Data mask."""
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

    def to_hdu(self, hdu_uncertainty='UNCERT',
               hdu_mask='MASK', unit_key='BUNIT',
               wcs_relax=True):
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

        Return
        ------
        `~astropy.fits.HDUList` :
            HDU storing all FrameData informations.
        """
        # TODO: Add history
        data = self.data.copy()
        header = fits.Header(self.header)
        if self.wcs is not None:
            header.extend(self.wcs.to_header(relax=wcs_relax),
                          useblanks=False, update=True)
        header[unit_key] = self.unit.to_string()
        hdul = fits.HDUList(fits.PrimaryHDU(data, header=header))

        if hdu_uncertainty is not None and self.uncertainty is not None:
            uncert = self.uncertainty
            uncert_unit = self.uncert_unit.to_string()
            uncert_h = fits.Header()
            uncert_h[unit_key] = uncert_unit
            hdul.append(fits.ImageHDU(uncert, header=uncert_h,
                                    name=hdu_uncertainty))

        if hdu_mask is not None and self.mask is not None:
            mask = self.mask
            hdul.append(fits.ImageHDU(mask, name=hdu_mask))

        return hdul
