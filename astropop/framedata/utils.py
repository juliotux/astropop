import six
import numbers
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, NDData

from .memmap import MemMapArray
from ..logger import logger
from .framedata import FrameData
from .compat import imhdus


__all__ = ['check_framedata']


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