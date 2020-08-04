# Licensed under a 3-clause BSD style license - see LICENSE.rst

from skimage.feature import register_translation
from skimage import transform
from scipy.ndimage import fourier_shift
import numpy as np
try:
    import astroalign
except Exception:
    astroalign = None

from ..logger import logger


# TODO: wrapping up with CCDPROC transform_image for translating


def translate(image, shift, subpixel=True, cval=0):
    """Translate an image by (dy, dx) using Scikit AffineTransform.
    
    Parameters
    ----------
    image : `~nnumpy.ndarray`
        2D image data to be translated.
    shift : `tuple (dx, dy)`
        The shift along the axes. Shift should contain one value for each axis.
    subpixel : `boolean (optional)`
        Consider the CCD subpixels on the translation.
        Default = True
    cval : `float (optional)`
        Value to fill empty pixels after shift.
        Default = 0
        
    Returns
    -------
    image_new : `~nnumpy.ndarray`
        2D shifted image.
    """
    # dy, dx = shift
    # translated = (dx, dy)

    tform = transform.AffineTransform(translation=shift)
    return transform.warp(image, tform, mode='constant', cval=cval)


def create_fft_shift_list(image_list):
    """Use fft to calculate the shifts between images in a list.

    Return a set os (x, y) shift pairs.
    """
    shifts = [(0.0, 0.0)]*len(image_list)
    for i in range(len(image_list)-1):
        shifts[i+1], _, _ = register_translation(image_list[0].data,
                                                 image_list[i+1].data)

    return shifts


def create_chi2_shift_list(image_list):
    """Calculate the shift between images using chi2 minimization.

    Uses image_registration.chi2_shift module.
    """
    from image_registration import chi2_shift

    shifts = [(0.0, 0.0)]*len(image_list)
    for i in range(len(image_list)-1):
        im = image_list[i+1]
        err = np.nanstd(im)
        dx, dy, _, _ = chi2_shift(image_list[0], im, err)
        shifts[i+1] = (-dx, -dy)

    return shifts


def apply_shift(image, shift, method='fft', subpixel=True, footprint=False,
                logger=logger):
    """Apply a shifts of (dx, dy) to a list of images.

    Parameters:
        image : ndarray_like
            The image to be shifted.
        shift: array_like
            shift to be applyed (dx, dy)
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


def apply_shift_list(image_list, shift_list, method='fft',
                     logger=logger):
    """Apply a list of (x, y) shifts to a list of images.

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
    return [apply_shift(i, s, method=method, logger=logger)
            for i, s in zip(image_list, shift_list)]


def hdu_shift_images(hdu_list, method='fft', register_method='asterism',
                     footprint=False, logger=logger):
    """Calculate and apply shifts in a set of ccddata images.

    The function process the list inplace. Original data altered.

    methods:
        - "asterism" : align images using asterism matching (astroalign)
        - "chi2" : align images using chi2 minimization (image_registration)
        - "fft" : align images using fourier transform correlation (skimage)
    """
    if method == "asterism":
        logger.info("Registering images with astroalign.")
        if astroalign is None:
            raise RuntimeError("astroaling module not available.")
        im0 = hdu_list[0].data
        for i in hdu_list[1:]:
            transf, _ = astroalign.find_transform(i.data, im0)
            i.data = astroalign.apply_transform(transf, i.data, im0)
            if footprint:
                i.footprint = astroalign.apply_transform(transf,
                                                         np.ones(i.data.shape,
                                                                 dtype=bool),
                                                         im0)
            s_method = 'similarity_transform'
    else:
        if method == 'chi2':
            shifts = create_chi2_shift_list([ccd.data for ccd in hdu_list])
        else:
            shifts = create_fft_shift_list([ccd.data for ccd in hdu_list])
        logger.info(f"Aligning CCDData with shifts: {shifts}")
        for ccd, shift in zip(hdu_list, shifts):
            if method == 'fft':
                s_method = method
            else:
                s_method = 'simple'
            ccd.data = apply_shift(ccd.data, shift, method=s_method,
                                   logger=logger)
            sh_string = [str(i) for i in shift]
            ccd.header['hierarch astropop register_shift'] = ",".join(sh_string)
            if footprint:
                ccd.footprint = apply_shift(np.ones_like(ccd.data, dtype=bool),
                                            shift, method='simple',
                                            logger=logger)
    for i in hdu_list:
        i.header['hierarch astropop registered'] = True
        i.header['hierarch astropop register_method'] = method
        i.header['hierarch astropop transform_method'] = s_method

    return hdu_list
