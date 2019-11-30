# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Custom frame class to support memmapping."""

# Important: We reimplemented and renamed astropy's frame to better
# handle disk memmap and an easier unit/uncertainty workflow

import six
from pathlib import Path
import os
import shutil
import numpy as np
import copy
from tempfile import mkdtemp, mkstemp
from astropy import units as u
from astropy.io import fits
from astropy.nddata import StdDevUncertainty, NDUncertainty, CCDData, NDData
from astropy.nddata.ccddata import _generate_wcs_and_update_header

from .py_utils import mkdir_p
from .memmap import MemMapArray
from .logger import logger


__all__ = ['FrameData', 'shape_consistency', 'unit_consistency',
           'check_framedata', 'setup_filename', 'framedata_read_fits',
           'framedata_to_hdu', 'extract_units']


# TODO: FrameData initializers for CCDData and HDUList with functions


_unsupport_fits_open_keywords = {
    'do_not_scale_image_data': 'Image data must be scaled.',
    'scale_back': 'Scale information is not preserved.'
}


imhdus = (fits.ImageHDU, fits.PrimaryHDU, fits.CompImageHDU,
          fits.StreamingHDU)


def check_framedata(data, ext=0, ext_mask='MASK', ext_uncert='UNCERT',
                    bunit='BUNIT', uunit=None, logger=logger):
    """Check if a data is a valid CCDData or convert it."""
    if isinstance(data, (FrameData, CCDData)):
        return FrameData(data)
    elif isinstance(data, NDData):
        ccd = FrameData(data.data, mask=data.mask,
                        uncertainty=data.uncertainty,
                        meta=data.meta, unit=data.unit or u.Unit(bunit))
        ccd.filename = None
        return ccd
    else:
        if isinstance(data, fits.HDUList):
            logger.debug("Extracting FrameData from ext {} of HDUList"
                         .format(ext))
            # TODO: Implement this
            raise NotImplementedError
            # return hdulist2framedata(data, ext, ext_mask, ext_uncert,
            #                          bunit=bunit, uunit=uunit)
        elif isinstance(data, six.string_types):
            logger.debug("Loading FrameData from {} file".format(data))
            try:
                ccd = FrameData.read_fits(data, hdu=ext)
                ccd.filename = data
                return ccd
            except ValueError:
                raise NotImplementedError
                # TODO: implement this
                # data = fits.open(data)
                # return hdulist2framedata(data, ext, ext_mask, ext_uncert,
                #                          bunit=bunit, uunit=uunit)
        elif isinstance(data, imhdus):
            logger.debug("Loading FrameData from {} HDU".format(data))
            raise NotImplementedError
            # TODO: implement this
            # hdu2framedata(data, bunit=bunit)
        else:
            raise ValueError(f'{data.__class__.__name__}'
                             ' is not a valid FrameData data type.')
    return data


def shape_consistency(data=None, uncertainty=None, mask=None):
    """Check shape consistency across data, uncertaitny and mask"""
    if data is None and uncertainty is not None:
        raise ValueError('Uncertainty set for an empty data.')
    if data is None and mask is not None:
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
            uncertainty = np.zeros(dshape)*uncertainty
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
        if not dunit is unit:
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
    elif u.Unit(data_unit) is not u.Unit(uncertainty_unit):
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

    The main difference from Astropy's CCDData is the memmapping itself.
    However it handles uncertainties in a totally different way. It stores only
    StdDev uncertainty arrays. It also stores the unit.

    Parameters:
    -----------
    - data : array_like or `astropy.units.Quantity`
        The main data values. If `Quantity`, unit will be set automatically.
    - unit : `astropy.units.Unit` or string (optional)
        The data unit. Must be `astropy.units.Unit` compilant.
    - dtype : string or `numpy.dtype` (optional)
        Mandatory dtype of the data.
    - uncertainty : array_like or `astropy.nddata.Uncertanty` or None \
                    (optional)
        Uncertainty of the data.
    - u_unit : `astropy.units.Unit` or string (optional)
        Unit of uncertainty of the data. If None, will be the same of the
        data.
    - u_dtype : string or `numpy.dtype` (optional)
        Mandatory dtype of uncertainty.
    - mask : array_like or None (optional)
        Frame mask.
    - m_dtype : string or `numpy.dtype` (optional)
        Mandatory dtype of mask. Default `bool`
    - wcs : dict, `astropy.fits.Header` or `astropy.wcs.WCS` (optional)
        World Coordinate System of the image.
    - meta or header: dict or `astropy.fits.Header` (optional)
        Metadata (header) of the frame. If both set, they will be merged.
        `header` priority.
    - cache_folder : string, `pathlib.Path` or None (optional)
        Place to store the cached FrameData
    - cache_filename : string, `pathlib.Path` or None (optional)
        Base file name to store the cached FrameData.
    - use_memmap_backend : bool (optional)
        True if enable memmap in constructor.
    """
    # TODO: Math operations (__add__, __subtract__, etc...)
    # TODO: Complete reimplement the initializer
    _memmapping = False
    _data = None
    _mask = None
    _unct = None
    _wcs = None
    _meta = None

    def __init__(self, data, unit=None, dtype=None,
                 uncertainty=None, u_unit=None, u_dtype=None,
                 mask=None, m_dtype=bool,
                 wcs=None, meta=None, header=None,
                 cache_folder=None, cache_filename=None, use_memmap_backend=False):
        self.cache_folder = cache_folder
        self.cache_filename = cache_filename

        # Setup MemMapArray instances
        cache_file = setup_filename(self, self.cache_folder, self.cache_filename)
        self._data = MemMapArray(None, cache_file + '.data')
        self._unct = MemMapArray(None, cache_file + '.unct')
        self._mask = MemMapArray(None, cache_file + '.mask')

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

        # TODO: Handle wcs creation

        if meta is not None and header is not None:
            header = dict(header)
            meta = dict(meta)
            self._meta = header
            self._meta.update(meta)
        elif meta is not None:
            self._meta = dict(meta)
        elif header is not None:
            self._meta = dict(meta)
        else:
            self._meta = dict()

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
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data.reset_data(value)

    @property
    def uncertainty(self):
        return self._unct

    @uncertainty.setter
    def uncertainty(self, value):
        _, value, _ = shape_consistency(self.data, value, None)
        unit_consistency(self.data, value)
        self._unct.reset_data(value)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask.reset_data(value)

    def enable_memmap(self, filename=None, cache_folder=None):
        """Enable array file memmapping."""
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
        uncert_h[unit_key] = uncert_unit
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
            if unit in unc_header.keys():
                val = unc_header[unit].strip()
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
                      uncertainty=uncertainty, u_unit=uunit,
                      mask=mask, use_memmap_backend=use_memmap_backend)
    hdul.close()

    return frame
