# Licensed under a 3-clause BSD style license - see LICENSE.rst

from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
import numpy as np
from astropy.wcs import WCS

from ..logger import logger


def create_fft_shift_list(image_list):
    """Use fft to calculate the shifts between images in a list.

    Return a set os (y,x) shift pairs.
    """
    shifts = [(0.0, 0.0)]*len(image_list)
    for i in range(len(image_list)-1):
        shifts[i+1], err, diff = register_translation(image_list[0].data,
                                                      image_list[i+1].data)

    return shifts


def apply_shift(image, shift, method='fft'):
    """Apply a list of (y,x) shifts to a list of images.

    Parameters:
        image : ndarray_like
            The image to be shifted.
        shift : array_like
            (y,x) shift pais, like the ones created by create_fft_shift_list.
        method : string
            The method used for shift images. Can be:
            - 'fft' -> scipy fourier_shift
            - 'wcs' -> astropy reproject_interp

    Return the shifted images.
    """
    # wcs method is used to avoid problems with fft transform in problematic
    # images
    if method == 'wcs':
        from reproject import reproject_interp
        cr = image.shape[0]/2

        w = WCS(naxis=2)
        w.wcs.crpix = [cr, cr]
        w.wcs.cdelt = [1, 1]
        w.wcs.crval = [0, 0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        w2 = WCS(naxis=2)
        w2.wcs.crpix = [cr - shift[1], cr - shift[0]]
        w2.wcs.cdelt = [1, 1]
        w2.wcs.crval = [0, 0]
        w2.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        return reproject_interp((image, w), w2, shape_out=image.shape)[0]

    # Shift with fft, much more precise and fast
    elif method == 'fft':
        nimage = fourier_shift(np.fft.fftn(image), np.array(shift))
        return np.fft.ifftn(nimage).real.astype(image.dtype)

    else:
        raise ValueError('Unrecognized shift image method.')


def apply_shift_list(image_list, shift_list, method='fft'):
    """Apply a list of (y,x) shifts to a list of images.

    Parameters:
        image_list : ndarray_like
            A list with the images to be shifted.
        shift_list : array_like
            A list with (y,x)shift pairs, like the ones created by
            create_fft_shift_list.
        method : string
            The method used for shift images. Can be:
            - 'fft' -> scipy fourier_shift
            - 'wcs' -> astropy reproject_interp

    Return a new image_list with the shifted images.
    """
    return [apply_shift(i, s, method=method) for i, s in zip(image_list,
                                                             shift_list)]


def hdu_shift_images(hdu_list, method='fft'):
    """Calculate and apply shifts in a set of ccddata images.

    The function process the list inplace. Original data altered.
    """
    shifts = create_fft_shift_list([ccd.data for ccd in hdu_list])
    logger.info("Aligning CCDData with shifts: {}".format(shifts))
    for ccd, shift in zip(hdu_list, shifts):
        ccd.data = apply_shift(ccd.data, shift, method=method)

    return hdu_list
