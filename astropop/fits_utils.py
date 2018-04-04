# Licensed under a 3-clause BSD style license - see LICENSE.rst

import six
from astropy.io import fits

__all__ = ['imhdus', 'check_header_keys', 'check_hdu', 'read_fits',
           'write_fits']

imhdus = (fits.ImageHDU, fits.PrimaryHDU, fits.CompImageHDU,
          fits.StreamingHDU)


class IncompatibleHeadersError(ValueError):
    """When 2 image header are not compatible"""


def check_header_keys(image1, image2, keywords=[]):
    """Compare some header keys from 2 images and check if the have equal
    values."""
    image1 = check_hdu(image1)
    image2 = check_hdu(image2)
    for i in keywords:
        if i in image1.header.keys():
            if i in image2.header.keys():
                v1 = image1.header[i]
                v2 = image2.header[i]
                if v1 != v2:
                    raise IncompatibleHeadersError('Keyword {} have different '
                                                   'values for images 1 and 2:'
                                                   '\n{}\n{}'.format(i,
                                                                     v1, v2))
            else:
                raise IncompatibleHeadersError('Image 2 do not have Keyword {}'
                                               .format(i))
        else:
            raise IncompatibleHeadersError('Image 1 do not have Keyword {}'
                                           .format(i))
    return True


def check_hdu(data):
    """Check if a data is a valid ImageHDU type and convert it."""
    if not isinstance(data, imhdus):
        if isinstance(data, fits.HDUList):
            data = data[0]
        elif isinstance(data, six.string_types):
            data = fits.open(data)[0]
        elif isinstance(data, np.ndarray):
            data = fits.PrimaryHDU(data)
        else:
            raise ValueError('The given data is not a valid CCDData type.')
    return data
