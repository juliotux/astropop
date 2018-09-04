# Licensed under a 3-clause BSD style license - see LICENSE.rst

from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift as scipy_shift
from scipy.signal import correlate2d
import numpy as np
try:
    import astroalign
except Exception:
    astroalign = None

from ..logger import logger


def translate(image, shift, subpixel=True, cval=0):
    """Translate an image by (dy, dx) using scipy.

    Based on stsci.image.translation algorithm

    cval = value to fill empty pixels after shift.
    """
    # for subpixel, a correlation kernel need translation
    if subpixel:
        rot = 0
        dy, dx = shift
        dx, dy = -dx, -dy
        if dx >= 0 and dy >= 0:
            rot = 2
        elif dx >= 0 and dy < 0:
            rot = 1
        elif dx < 0 and dy >= 0:
            rot = 3
        elif dx <0 and dy < 0:
            rot = 0
        dx, dy = np.abs([dx, dy])
        if rot % 2 != 0:
            dx, dy = dy, dx

        nim = np.rot90(image, rot)
        nim = scipy_shift(nim, (dy, dx), mode='constant', cval=cval)

        # correlation kernel to fix subpixel shifting
        x, y = dx % 1.0, dy % 1.0
        kernel = np.array([[x*y    , (1-x)*y    ],
                           [(1-y)*x, (1-y)*(1-x)]])
        nim = correlate2d(nim, kernel, mode='full', fillvalue=cval)
        return np.rot90(nim, -rot % 4).astype(image.dtype)[:-1, :-1]

    return scipy_shift(image, shift, mode='constant', cval=cval)


def create_fft_shift_list(image_list):
    """Use fft to calculate the shifts between images in a list.

    Return a set os (y, x) shift pairs.
    """
    shifts = [(0.0, 0.0)]*len(image_list)
    for i in range(len(image_list)-1):
        shifts[i+1], err, diff = register_translation(image_list[0].data,
                                                      image_list[i+1].data)

    return shifts


def apply_shift(image, shift, method='fft', subpixel=True, footprint=False):
    """Apply a shifts of (dy, dx) to a list of images.

    Parameters:
        image : ndarray_like
            The image to be shifted.
        shift: array_like
            shift to be applyed (dy, dx)
        method : string
            The method used for shift images. Can be:
            - 'fft' -> scipy fourier_shift
            - 'simple' -> simples translate using scipy

    Return the shifted images.
    """
    # Shift with fft, much more precise and fast
    if method == 'fft':
        nimage = fourier_shift(np.fft.fftn(image), np.array(shift))
        nimage = np.fft.ifftn(nimage).real.astype(image.dtype)
        if footprint:
            foot = np.ones(nimage.shape)
            foot = translate(foot, shift, subpixel=True, cval=0)
            return nimage, foot
        else:
            return nimage

    elif method == 'simple':
        nimage = translate(image, shift, subpixel=subpixel, cval=0)
        if footprint:
            foot = np.ones(nimage.shape)
            foot = translate(foot, shift, subpixel=subpixel, cval=0)
            return nimage, foot
        else:
            return nimage

    else:
        raise ValueError('Unrecognized shift image method.')


def apply_shift_list(image_list, shift_list, method='fft'):
    """Apply a list of (y, x) shifts to a list of images.

    Parameters:
        image_list : ndarray_like
            A list with the images to be shifted.
        shift_list : array_like
            A list with (x, y)shift pairs, like the ones created by
            create_fft_shift_list.
        method : string
            The method used for shift images. Can be:
            - 'fft' -> scipy fourier_shift
            - 'simple' -> simples translate using scipy

    Return a new image_list with the shifted images.
    """
    return [apply_shift(i, s, method=method) for i, s in zip(image_list,
                                                              shift_list)]


def hdu_shift_images(hdu_list, method='fft'):
    """Calculate and apply shifts in a set of ccddata images.

    The function process the list inplace. Original data altered.

    methods:
        - "asterism" : align images using asterism matching (astroalign)
        - "fft" : align images using fourier transform correlation (skimage)
    """
    if method == "asterism":
        logger.info("Registering images with astroalign.")
        if astroalign is None:
            raise RuntimeError("astroaling module not available.")
        im0 = hdu_list[0].data
        for i in hdu_list[1:]:
            i.data = astroalign.register(i.data, im0)
            i.header['hierarch astropop astroalign'] = True
    else:
        shifts = create_fft_shift_list([ccd.data for ccd in hdu_list])
        logger.info("Aligning CCDData with shifts: {}".format(shifts))
        for ccd, shift in zip(hdu_list, shifts):
            ccd.data = apply_shift(ccd.data, shift, method=method)
            ccd.header['hierarch astropop registered'] = True

    return hdu_list
