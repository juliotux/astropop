'''Wrapper from ccdproc package.'''
import six
import ccdproc
from ccdproc import CCDData
from astropy import units as u
from astropy.io import fits

# from ..logger import logger

mem_limit = 1e8


def read_fits(fname):
    try:
        return ccdproc.CCDData.read(fname)
    except ValueError:
        # assume if not gain corrected, the unit is adu
        return ccdproc.CCDData.read(fname, unit='adu')


def save_fits(ccd, fname):
    ccd.write(fname)


_ccd_procces_keys = ['oscan', 'trim', 'error', 'master_bias', 'dark_frame',
                     'master_flat', 'bad_pixel_mask', 'gain', 'readnoise',
                     'oscan_median', 'oscan_model', 'min_value',
                     'dark_exposure', 'data_exposure', 'exposure_key',
                     'exposure_unit', 'dark_scale', 'gain_corrected',
                     'add_keyword']


def check_ccddata(image):
    """Check if a image is a CCDData. If not, try to convert it to CCDData."""
    if not isinstance(image, ccdproc.CCDData):
        if isinstance(image, six.string_types):
            return read_fits(image)
        elif isinstance(image, (fits.HDUList)):
            return CCDData(image[0].data, meta=image[0].header)
        elif isinstance(image, (fits.ImageHDU, fits.PrimaryHDU,
                                fits.CompImageHDU)):
            return CCDData(image.data, meta=image.header)
        else:
            raise ValueError('image type {} not supported'.format(type(image)))

    return image


def process_image(ccd, gain_key=None, readnoise_key=None,
                  *args, **kwargs):
    ccd = check_ccddata(ccd)
    nkwargs = {}
    for i in kwargs.keys():
        if i in _ccd_procces_keys:
            nkwargs[i] = kwargs.get(i)

    for i in ['master_bias', 'master_flat', 'dark_frame', 'badpixmask']:
        if nkwargs.get(i, None):
            nkwargs[i] = check_ccddata(nkwargs[i])

    if gain_key is not None:
        nkwargs['gain'] = float(ccd.header[gain_key])*u.electron/u.adu

    if readnoise_key is not None:
        if isinstance(ccd, ccdproc.CCDData):
            nkwargs['readnoise'] = float(ccd.header[readnoise_key])*u.electron
        else:
            nkwargs['readnoise'] = float(fits.getval(gain_key))*u.electron

    if kwargs.get('master_bias', None):
        kwargs['master_bias'] = check_ccddata(kwargs['master_bias'])
    if kwargs.get('master_flat', None):
        kwargs['master_flat'] = check_ccddata(kwargs['master_flat'])
    if kwargs.get('dark_frame', None):
        kwargs['dark_frame'] = check_ccddata(kwargs['dark_frame'])
    if kwargs.get('badpixmask', None):
        kwargs['badpixmask'] = check_ccddata(kwargs['badpixmask'])

    # logger.debug('ccdproc kwargs:{}'.format(nkwargs))

    ccd = ccdproc.ccd_process(ccd, *args, **nkwargs)

    if gain_key is not None:
        ccd.meta[gain_key] = 1.0

    return ccd
