# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Custom frame class to support memmapping."""

# Important: We reimplemented and renamed astropy's frame to better
# handle disk memmap and an easier unit/uncertainty workflow

from pathlib import Path
import os
import shutil
import numpy as np
import copy
from tempfile import mkdtemp, mkstemp
from astropy import units as u
from astropy.io.fits import Header
from astropy.nddata import StdDevUncertainty, NDUncertainty, CCDData

from .py_utils import mkdir_p


__all__ = ['FrameData', 'create_array_memmap', 'delete_array_memmap',
           'ensure_bool_mask', 'setup_filename']


def ensure_bool_mask(value):
    """Ensure a mask value is bool"""
    if hasattr(value, 'dtype'):
        # If bool, just return
        if np.dtype(value.dtype) is np.dtype(np.bool):
            return value
    
    # handle memmap
    if isinstance(value, np.memmap):
        filename = Path(value.filename)
        value = delete_array_memmap(value)
    else:
        filename = None

    value = np.array(value).astype('bool')

    if filename is not None:
        value = create_array_memmap(filename.open('w'), value)

    return value


def create_array_memmap(filename, data, dtype=None):
    """Create a memory map to an array data."""
    if data is None:
        return

    dtype = dtype or data.dtype
    shape = data.shape
    if data.ndim > 0:
        memmap = np.memmap(filename, mode='w+', dtype=dtype, shape=shape)
        memmap[:] = data[:]
    else:
        memmap = data
    return memmap


def delete_array_memmap(memmap, read=True):
    """Delete a memmap and read the data to a np.ndarray"""
    if memmap is None:
        return

    if read:
        data = np.array(memmap[:])
    else:
        data = None
    name = memmap.filename
    del memmap
    os.remove(name)
    return data


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
    
    The main difference from Astropy's CCDData is the memmapping itself. However
    It handles uncertainties in a totally different way. It stores only
    StdDev uncertainty arrays. It also stores the unit.

    Parameters:
        - data : array_like, `astropy.nddata.CCDData`, or \
                 `astropy.units.Quantity`
            The main data values.
        - unit : `astropy.units.Unit` or string (optional)
            The data unit. Must be `astropy.units.Unit` compilant.
        - uncertainty : array_like or `astropy.nddata.Uncertanty` or None \
                        optional
            Uncertainty of the data.
        - uncertainty_unit : `astropy.units.Unit` or string (optional)

        - mask : array_like or None (optional)
            Frame mask.
        - cache_folder : string, `pathlib.Path` or None (optional)
            Place to store the cached FrameData
        - cache_filename : string, `pathlib.Path` or None (optional)
            Base file name to store the cached FrameData.
        - use_memmap_backend : bool (optional)
            True if enable memmap in constructor.
    """
    # TODO: Math operations (__add__, __subtract__, etc...)
    # TODO: handle masked arrays
    _memmapping = False
    _data = None
    _data_unit = None
    _mask = None
    _unct = None
    _unct_unit = None
    _wcs = None
    _meta = Header()

    def __init__(self, data, *args, **kwargs):
        if data is None:
            raise TypeError('Data cannot be None')
        elif isinstance(data, FrameData):
            meta = copy.copy(data.meta)
            uncertainty = copy.copy(data.uncertainty)
            unit = copy.copy(data.unit)
            uncert_unit = copy.copy(data.uncert_unit)
            mask = copy.copy(data.mask)
            wcs = copy.copy(data.wcs)
        elif isinstance(data, CCDData):
            meta = copy.copy(CCDData.meta)
            uncertainty = copy.copy(CCDData.uncertainty)
            uncert_unit = uncertainty.unit
            unit = copy.copy(CCDData.unit)
            mask = copy.copy(CCDData.mask)
            data = data.data.copy()
            wcs = copy.copy(data.wcs)
        elif isinstance(data, u.Quantity):
            unit = copy.copy(data.unit)
            data = copy.copy(data.value)
            meta = None
            uncertainty = None
            uncert_unit = None
            mask = None
            wcs = None
        else:
            unit = None
            meta = None
            uncertainty = None
            uncert_unit = None
            mask = None
            wcs = None

        self.data = data

        if unit is not None:
            if kwargs.get('unit', None) is not None:
                raise ValueError('Data with unit defined and manual unit specified.'
                                 ' Incompatible behavior.')
            self.unit = unit
        else:
            # Mimic astropys CCDData behavior?
            raise TypeError('Unit cannot be None. If unit not wanted, set \'\'')

        if 'meta' not in kwargs:
            kwargs['meta'] = kwargs.pop('header', None)
        if 'header' in kwargs:
            raise ValueError("can't have both header and meta.")
        if kwargs.get('meta', None) is not None and meta is not None:
            raise ValueError('Meta already set by data.')
        
        meta = meta or kwargs.get('meta', None)
        if meta is not None:
            self._meta = Header()

        self.cache_folder = kwargs.pop('cache_folder', None)
        self.cache_filename = kwargs.pop('cache_filename', None)

        self._memmapping = False
        memmap = kwargs.pop('use_memmap_backend', None)
        if memmap:
            self.enable_memmap()

        if uncertainty is not None:
            if kwargs.get('uncertainty', None) is not None:
                raise ValueError('Data with uncertainty defined and manual '
                                 ' uncertainty specified.'
                                 ' Incompatible behavior.')
        else:
            uncertainty = kwargs.pop('uncertainty', None)
        if uncertainty is not None:
            self.uncertainty = uncertainty

        if uncert_unit is not None:
            if kwargs.get('uncertainty_unit', None) is not None:
                raise ValueError('Data with uncertainty unit defined and manual '
                                 ' uncertainty_unit specified.'
                                 ' Incompatible behavior.')
        else:
            uncert_unit = kwargs.pop('uncertain_unit', None)
        if uncert_unit is not None:
            uncert_unit = u.Unit(uncert_unit)
            if self._unct_unit is not None and \
               uncert_unit is not self._unct_unit:
                raise ValueError('Uncertainty unit already set and'
                                 ' incompatible.')
            self.uncert_unit = uncert_unit

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def wcs(self):
        return self._wcs

    @wcs.setter
    def wcs(self, value):
        self._wcs = value

    @property
    def meta(self):
        return self._meta
    
    @meta.setter
    def meta(self, value):
        self._meta = Header(value)

    @property
    def unit(self):
        return self._data_unit

    @unit.setter
    def unit(self, value):
        value = u.Unit(value)
        self._data_unit = value

    @property
    def uncert_unit(self):
        return self._unct_unit or self._unit

    @uncert_unit.setter
    def uncert_unit(self, value):
        value = u.Unit(value)
        self._unct_unit = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, u.Quantity):
            self.unit = value.unit
            value = value.value
        elif isinstance(value, CCDData):
            # From a CCDData, just insert data and unit
            self.unit = value.unit
            value = value.data
        value = np.array(value)  # ensure a array

        if self._memmapping:
            if self._data.shape != value.shape or \
               self._data.dtype != value.dtype:
                name = self._data.filename
                delete_array_memmap(self._data)
                self._data = create_array_memmap(name, value)
            else:
                self._data[:] = value[:]
        else:
            self._data = value

    @data.deleter
    def data(self):
        if self._memmapping:
            name = self._data.filename
            dirname = os.path.dirname(name)
            delete_array_memmap(self._data, read=False)
            os.remove(name)

            if len(os.listdir(dirname)) == 0:
                shutil.rmtree(dirname)
        else:
            del self._data

    @property
    def uncertainty(self):
        return self._unct

    @uncertainty.setter
    def uncertainty(self, value):
        if isinstance(value, NDUncertainty):
            if isinstance(value, StdDevUncertainty):
                self.unct_unit = value.unit
                value = value.array
            else:
                raise ValueError('Only StdDevUncertainty supported.')
        
        if isinstance(value, u.Quantity):
            self.unct_unit = value.unit
            value = value.value

        if value is not None:
            value = np.array(value)
            if value.shape == ():
                narr = np.zeros(self.shape, dtype=value.dtype)
                narr[:] = value
                value = narr
            elif value.shape != self.shape:
                raise ValueError(f'Uncertainty with shape {value.shape}'
                                 'incompatible with FrameData with shape '
                                 '{self.shape}.')
        else:
            delete_array_memmap(self._unct, rad=False)
            self._unct = None
            return

        if self._memmapping:
            if self._unct is None:
                fname = setup_filename(self) + '.uncert'
                self._unct = create_array_memmap(fname, value)
            elif self._unct.shape != value.shape or \
                 self._unct.dtype != value.dtype:
                name = self._unct.filename
                delete_array_memmap(self._unct)
                self._unct = create_array_memmap(name, value)
            else:
                self._unct[:] = value[:]
        else:
            self._unct = value
    
    @uncertainty.deleter
    def uncertainty(self):
        if isinstance(self._unct, np.memmap):
            name = self._uncertainty.filename
            dirname = os.path.dirname(name)
            delete_array_memmap(self._uncertainty, read=False)
            os.remove(name)

            if len(os.listdir(dirname)) == 0:
                shutil.rmtree(dirname)
        else:
            del self._uncertainty

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        if value is not None:
            value = np.array(value)
            if value.shape == ():
                narr = np.zeros(self.shape, dtype=bool)
                narr[:] = value
                value = narr
            elif value.shape != self.shape:
                raise ValueError(f'Mask with shape {value.shape} incompatible '
                                 'with FrameData with shape {self.shape}.')
        else:
            delete_array_memmap(self._mask, read=False)
            self._mask = None
            return

        value = ensure_bool_mask(value)

        if self._mask is None and self._memmapping:
            delete_array_memmap(self._mask, read=False)
            name = setup_filename(self, self.cache_folder,
                                  self.cache_filename) + '.mask'
            self._mask = create_array_memmap(name, value, dtype=bool)
        else:
            delete_array_memmap(self._mask, read=False)
            self._mask = value

    @mask.deleter
    def mask(self):
        if isinstance(self._mask, np.memmap):
            name = self._mask.filename
            dirname = os.path.dirname(name)
            delete_array_memmap(self._mask, read=False)
            os.remove(name)

            if len(os.listdir(dirname)) == 0:
                shutil.rmtree(dirname)
        else:
            del self._mask

    def enable_memmap(self, filename=None, cache_folder=None):
        """Enable array file memmapping."""
        cache_file = setup_filename(self, cache_folder, filename)

        self._data = create_array_memmap(cache_file + '.data', self._data)
        self._mask = create_array_memmap(cache_file + '.mask', self._mask)
        self._unct = create_array_memmap(cache_file + '.unct', self._mask)

        self._memmapping = True

    def disable_memmap(self):
        """Disable frame file memmapping (load to memory)."""

        self._data = delete_array_memmap(self._data, read=True)
        self._mask = delete_array_memmap(self._mask, read=True)
        self._unct = delete_array_memmap(self._unct, read=True)
        self._memmapping = False
