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
from astropy.io import fits, registry
from astropy.nddata import StdDevUncertainty, NDUncertainty, CCDData
from astropy.nddata.ccddata import _generate_wcs_and_update_header

from .py_utils import mkdir_p
from .logger import logger


__all__ = ['FrameData', 'create_array_memmap', 'delete_array_memmap',
           'ensure_bool_mask', 'setup_filename', 'framedata_read_fits',
           'framedata_to_hdu']


_unsupport_fits_open_keywords = {
    'do_not_scale_image_data': 'Image data must be scaled.',
    'scale_back': 'Scale information is not preserved.'
}


imhdus = (fits.ImageHDU, fits.PrimaryHDU, fits.CompImageHDU,
          fits.StreamingHDU)


def ensure_bool_mask(value):
    """Ensure a mask value is bool"""
    if hasattr(value, 'dtype'):
        # If bool, just return
        if np.dtype(value.dtype) is np.dtype(np.bool):
            return value

    # handle memmap
    filename = None
    if isinstance(value, np.memmap):
        filename = Path(value.filename)
        value = delete_array_memmap(value)

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

    The main difference from Astropy's CCDData is the memmapping itself.
    However it handles uncertainties in a totally different way. It stores only
    StdDev uncertainty arrays. It also stores the unit.

    Parameters:
    -----------
    - data : array_like, `astropy.nddata.CCDData`, or \
                `astropy.units.Quantity`
        The main data values.
    - unit : `astropy.units.Unit` or string (optional)
        The data unit. Must be `astropy.units.Unit` compilant.
    - uncertainty : array_like or `astropy.nddata.Uncertanty` or None \
                    optional
        Uncertainty of the data.
    - uncertainty_unit : `astropy.units.Unit` or string (optional)
        Unit of uncertainty of the data. If None, will be the same of the
        data.
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
    # TODO: check data and uncertianty if they are numbers during set
    _memmapping = False
    _data = None
    _data_unit = None
    _mask = None
    _unct = None
    _unct_unit = None
    _wcs = None
    _meta = dict()

    def __init__(self, data, *args, **kwargs):
        self.cache_folder = kwargs.pop('cache_folder', None)
        self.cache_filename = kwargs.pop('cache_filename', None)

        self._memmapping = False
        memmap = kwargs.pop('use_memmap_backend', False)
        if memmap:
            self.enable_memmap()

        if data is None:
            raise TypeError('Data cannot be None')
        elif isinstance(data, FrameData):
            meta = copy.copy(data.meta)
            uncertainty = copy.copy(data.uncertainty)
            unit = copy.copy(data.unit)
            uncert_unit = copy.copy(data.uncert_unit)
            mask = copy.copy(data.mask)
            wcs = copy.copy(data.wcs)
            data = copy.copy(data.data)
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
                raise ValueError('Data with unit defined and manual unit '
                                 'specified. Incompatible behavior.')
            self.unit = unit
        elif kwargs.get('unit', None) is not None:
            self.unit = kwargs.pop('unit', None)
        else:
            # Mimic astropys CCDData behavior?
            raise ValueError('Unit cannot be None. If unit not wanted, set ""')

        if 'meta' not in kwargs:
            kwargs['meta'] = kwargs.pop('header', None)
        if 'header' in kwargs:
            raise ValueError("can't have both header and meta.")
        if kwargs.get('meta', None) is not None and meta is not None:
            raise ValueError('Meta already set by data.')

        meta = meta or kwargs.get('meta', None)
        if meta is not None:
            self._meta = dict(meta)

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
                raise ValueError('Data with uncertainty unit defined and'
                                 ' manual uncertainty_unit specified.'
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

        if wcs is not None:
            if kwargs.get('uncertainty_unit', None) is not None:
                raise ValueError('Data with wcs defined and'
                                 ' manual wcs specified.'
                                 ' Incompatible behavior.')
            else:
                wcs = kwargs.pop('wcs')
        if wcs is None:
            wcs = self._meta
        self.wcs = wcs

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def size(self):
        return self._data.size

    @property
    def wcs(self):
        return self._wcs

    @wcs.setter
    def wcs(self, value):
        self.header, wcs = _generate_wcs_and_update_header(value)
        self._wcs = wcs

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        self.header = value

    @property
    def header(self):
        return self._meta

    @header.setter
    def header(self, value):
        self._meta = dict(value)

    @property
    def unit(self):
        return self._data_unit

    @unit.setter
    def unit(self, value):
        value = u.Unit(value)
        self._data_unit = value

    @property
    def uncert_unit(self):
        if self._unct_unit is None:
            return self._data_unit
        return self._unct_unit

    @uncert_unit.setter
    def uncert_unit(self, value):
        print(f'unct_unit {value}')
        if value is None:
            self._unct_unit = value
        else:
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
            if self._data is None:
                name = setup_filename(self) + '.data'
                self._data = create_array_memmap(name, value)
            elif (self._data.shape != value.shape or
                  self._data.dtype != value.dtype):
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
                self.uncert_unit = value.unit
                value = value.array
            else:
                raise ValueError('Only StdDevUncertainty supported.')

        if isinstance(value, u.Quantity):
            self.uncert_unit = value.unit
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
            delete_array_memmap(self._unct, read=False)
            self._unct = None
            self.uncert_unit = None
            return

        if self._memmapping:
            if self._unct is None:
                fname = setup_filename(self) + '.uncert'
                self._unct = create_array_memmap(fname, value)
            elif (self._unct.shape != value.shape or
                  self._unct.dtype != value.dtype):
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
        self._unct = create_array_memmap(cache_file + '.unct', self._unct)
        self._memmapping = True

    def disable_memmap(self):
        """Disable frame file memmapping (load to memory)."""
        self._data = delete_array_memmap(self._data, read=True)
        self._mask = delete_array_memmap(self._mask, read=True)
        self._unct = delete_array_memmap(self._unct, read=True)
        self._memmapping = False

    def to_hdu(self, *args, **kwargs):
        f"""{framedata_to_hdu.__doc__}"""
        return framedata_to_hdu(self, *args, **kwargs)

    def read_fits(self, *args, **kwargs):
        f"""{framedata_read_fits.__doc__}"""
        if isinstance(self, FrameData):
            arg0 = []
        else:
            arg0 = [self]
        arg0.extend(args)
        return framedata_read_fits(*arg0, **kwargs)

    def write_fits(self, *args, **kwargs):
        f"""{framedata_write_fits.__doc__}"""
        return framedata_write_fits(self, *args, **kwargs)


def framedata_to_hdu(framedata, hdu_uncertainty='UNCERT',
                     hdu_mask='MASK', unit_key='BUNIT',
                     wcs_relax=True):
    """Generate an HDUList from this FrameData."""
    data = framedata.data.copy()
    header = fits.Header(framedata.header)
    if framedata.wcs is not None:
        header.extend(framedata.wcs.to_header(relax=wcs_relax),
                      useblanks=False, update=True)
    header[unit_key] = framedata.unit.to_string()
    hdul = fits.HDUList(fits.PrimaryHDU(data, header=header))

    if hdu_uncertainty is not None and framedata.uncertainty is not None:
        uncert = framedata.uncertainty
        uncert_unit = framedata.uncert_unit.to_string()
        uncert_h = fits.Header()
        uncert_h[unit_key] = uncert_h
        hdul.append(fits.ImageHDU(uncert, header=uncert_h,
                                  name=hdu_uncertainty))

    if hdu_mask is not None and framedata.mask is not None:
        mask = framedata.mask
        hdul.append(fits.ImageHDU(mask, name=hdu_mask))

    return hdul


def framedata_write_fits(framedata, filename, hdu_mask='MASK',
                         hdu_uncertainty='UNCERT', unit_key='BUNIT',
                         wcs_relax=True, **kwargs):
    """Write a framedata to a file."""
    hdul = framedata.to_hdu(hdu_uncertainty=hdu_uncertainty, hdu_mask=hdu_mask,
                            unit_key=unit_key, wcs_relax=wcs_relax)
    hdul.writeto(filename, **kwargs)


def framedata_read_fits(filename=None, hdu=0, unit='BUNIT',
                        hdu_uncertainty='UNCERT',
                        hdu_mask='MASK',
                        use_memmap_backend=False, **kwargs):
    f"""Create a FrameData from a FITS file.

    Parameters:
    -----------
    - filename : string, `pathlib.Path` or `astropy.io.fits.HDUList`
        File to be loaded. It can be passed in the form of a string or a
        `pathlib.Path` to be read by `astropy.io.fits`. If a
        `astropy.io.fits.HDUList is passed, it is directly loaded.
    - hdu : string or int (optional)
        Extension of the fits to be used as data provider. If 0, the reader
        will seek for the first HDU with valid data in file, skipping
        tables.
        Default: ``0``
    - unit : `astropy.units.Unit`, string or None (optional)
        Manual specifying the data unit. If a string is passed, first the
        code try to interpret the string as a unit. Except, the reader
        search for the string in header and try to colect the information
        from header. If None, the reader will seek in HDU header for unit
        using the default BUNIT key.
        Default: ``'BUNIT'``
    - hdu_uncertainty : string or int (optional)
        HDU containing the uncertainty data. Unit will be assigned
        according the `unit` argument. Only StdDevUncertainty is
        supported.
        Default: ``'UNCERT'``
    - hdu_mask : string or int (optional)
        HDU containing the mask data.
        Default: ``'MASK'``
    - kwargs :
        Keyword arguments to be passed to `astropy.io.fits`. The following
        keyowrds are not supported:
        {_unsupport_fits_open_keywords}
    """
    for key, msg in _unsupport_fits_open_keywords.items():
        if key in kwargs:
            prefix = f'unsupported keyword: {key}.'
            raise TypeError(' '.join([prefix, msg]))

    hdul = None
    if isinstance(filename, fits.HDUList):
        hdul = filename
    else:
        hdul = fits.open(filename, **kwargs)

    # Read data and header
    data_hdu = hdul[hdu]
    if data_hdu.data is None and hdu == 0:
        # Seek for first valid image data
        i = 1
        while i < len(hdul):
            if isinstance(hdul[i], imhdus):
                data_hdu = hdul[i]
                hdu = i
                break
    if data_hdu.data is None:
        raise ValueError('No valid image HDU found in fits file.')
    header = data_hdu.header

    # Unit
    dunit = None
    try:
        dunit = u.Unit(unit)
    except (TypeError, ValueError):
        if unit in header.keys():
            val = header[unit].strip()
            if val.lower() == 'adu':
                # fix problematic upper case adu
                val = val.lower()
            dunit = u.Unit(val)
        else:
            raise ValueError(f'Unit {unit} is not a valid astropy '
                             'unit or a valid unit key header.')

    # Uncertainty
    try:
        hdu_uncert = hdul[hdu_uncertainty]
    except KeyError:
        hdu_uncert = None
    if hdu_uncert is not None:
        if hdu_uncert == hdul[hdu]:
            raise ValueError('`hdu_uncertainty` and `hdu` cannot be the '
                             'same!')
        uncertainty = hdul[hdu_uncertainty]
        unc_header = uncertainty.header
        uncertainty = uncertainty.data
        uunit = None
        try:
            uunit = u.Unit(unit)
        except (TypeError, ValueError):
            if unit in header.keys():
                val = header[unit].strip()
                if val.lower() == 'adu':
                    # fix problematic upper case adu
                    val = val.lower()
                uunit = u.Unit(val)
            else:
                raise ValueError(f'Unit {unit} is not a valid astropy '
                                 'unit or a valid unit key header.')
    else:
        uncertainty = None
        uunit = None

    # Mask
    try:
        mask_hdu = hdul[hdu_mask]
    except KeyError:
        mask_hdu = None
    if mask_hdu is not None:
        if hdul[hdu_mask] == hdul[hdu]:
            raise ValueError('`hdu_mask` and `hdu` cannot be the '
                             'same!')
        mask = mask_hdu.data
    else:
        mask = None

    # WCS
    wcs = _generate_wcs_and_update_header(header)
    frame = FrameData(data_hdu.data, unit=dunit, wcs=wcs, meta=header,
                      uncertainty=uncertainty, uncertainty_unit=uunit,
                      mask=mask, use_memmap_backend=use_memmap_backend)
    hdul.close()

    return frame
