import six
import numbers
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, NDData
from .compat import imhdus, _unsupport_fits_open_keywords

from .memmap import MemMapArray
from ..logger import logger
from .framedata import FrameData
from .compat import imhdus


__all__ = ['check_framedata', 'framedata_read_fits',
           'framedata_write_fits']


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

    frame = FrameData(data_hdu.data, unit=dunit, meta=header,
                      uncertainty=uncertainty, u_unit=uunit,
                      mask=mask, use_memmap_backend=use_memmap_backend)
    hdul.close()

    return frame


def _fits_handle(data, logger=logger, **kwargs):
    if isinstance(data, fits.HDUList):
        ext = kwargs.pop('ext', 0)
        logger.debug("Extracting FrameData from ext {} of HDUList"
                     .format(ext))
        # TODO: Implement this
        raise NotImplementedError
        # return hdulist2framedata(data, ext, ext_mask, ext_uncert,
        #                          bunit=bunit, uunit=uunit)
    elif isinstance(data, six.string_types):
        ext = kwargs.pop('ext', 0)
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


def check_framedata(data, logger=logger, **kwargs):
    """Check if a data is a valid FrameData or convert it."""
    if isinstance(data, FrameData):
        return data
    elif isinstance(data, (NDData, CCDData)):
        ndata = {}
        for i in ['data', 'unit', 'meta', 'header', 'wcs',
                  'filename', 'uncertainty', 'mask']:
            if hasattr(data, i):
                ndata[i] = getattr(data, i)
        ndata['origin_filename'] = ndata.pop('filename', None)
        return FrameData(**ndata)
    elif isinstance(data, (MemMapArray, np.ndarray)):
        ndata = {}
        ndata['data'] = data
        if hasattr(data, 'unit'):
            ndata['unit'] = data.unit
        if isinstance(data, np.ma):
            ndata['mask'] = data.mask
        return FrameData(**ndata)
    elif isinstance(data, (u.Quantity, numbers.Number)):
        return FrameData(data)
    elif isinstance(data, (fits.HDUList, six.string_types, *imhdus)):
        return _fits_handle(data, **kwargs)
    else:
        raise ValueError(f'{data.__class__.__name__}'
                         ' is not a valid FrameData data type.')
