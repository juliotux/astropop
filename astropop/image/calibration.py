# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import numpy as np
import astroscrappy

from ..logger import logger
from .imarith import imarith
from ..framedata import check_framedata


__all__ = ['cosmics_lacosmic', 'gain_correct', 'subtract_bias',
           'subtract_dark', 'flat_correct', 'process_image']


# TODO: replace ccdproc functions by built-in, skiping units
# block_reduce = ccdproc.block_reduce
# block_replicate = ccdproc.block_replicate
# trim_image = ccdproc.trim_image
# subtract_overscan = partial(ccdproc.subtract_overscan,
#                             add_keyword='hierarch astropop'
#                             ' overscan_subtracted')


###############################################################################
# DONE
###############################################################################


def cosmics_lacosmic(frame, inplace=False, **lacosmic_kwargs):
    """
    Remove cosmic rays with LAcosmic. From astroscrappy package.

    Notes
    -----
    * Detailed information about the Laplacian Cosmic Ray Identification method
      can be found on Dokkum,P.G. (2001) - PASP 113, 1420 (2001)
      https://arxiv.org/pdf/astro-ph/0108003.pdf.

    * If `frame` is not a `~astropop.framedata.FrameData` instance,
      inplies in `inplace=False`, and a new `~astropop.framedata.FrameData`
      instance will be created.

    Parameters
    ----------
    frame: `~astropop.framedata.FrameData` compatible
        Values to perform the operation. `~astropy.units.Quantity`, numerical
        values and `~astropy.nddata.CCDData` are also suported.
    inplace: bool, optional
        If True, the operations will be performed inplace in the `frame`.
    logger: `~logging.Logger`
        Python logger to log the actions.

    Returns
    -------
    `~astropop.framedata.FrameData`
        New cosmic-rays corrected `FrameData` instance if not `inplace`,
        else the `image` `~astropop.framedata.FrameData` instance.
    """
    # As lacosmic removes and replace the cosmics pixels, no need to
    # update the mask
    _, dat = astroscrappy.detect_cosmics(frame.data, **lacosmic_kwargs)

    if inplace:
        ccd = frame
    else:
        ccd = frame.copy()

    ccd.data = dat
    # Do not update mask, astroscrappy replace the pixels
    # ccd.mask &= mask
    ccd.header['hierarch astropop lacosmic'] = True
    return ccd


def gain_correct(image, gain, gain_unit=None, inplace=False):
    """
    Process the gain correction of an image.

    Notes
    -----
    * The gain is implemented as a multiplier of the original image.

    * If ``image`` is not a `~astropop.framedata.FrameData` instance,
      inplies in ``inplace=False``, and a new `~astropop.framedata.FrameData`
      instance will be created.

    Parameters
    ----------
    image : `~astropop.framedata.FrameData` compatible
        Values to perform the operation. `~astropy.units.Quantity`, numerical
        values and `~astropy.nddata.CCDData` are also suported.
    gain : float, `~astropy.units.Quantity` ou `~astropop.math.QFloat`
        Gain to be applied on the image numerical values.
    gain_unit : str, optional
        Unit of the applied gain.
    inplace : bool, optional
        If True, the operations will be performed inplace in the `image`.

    Returns
    -------
    `~astropop.framedata.FrameData`:
        New gain corrected `FrameData` instance if not ``inplace``, else the
        ``image`` `~astropop.framedata.FrameData` instance.
    """
    # TODO: handle units
    nim = imarith(image, gain, '*', inplace=False)
    nim.header['hierarch astropop gain_corrected'] = True
    nim.header['hierarch astropop gain_corrected_value'] = gain
    # nim.header['hierarch astropop gain_corrected_unit'] = str(gain_unit)

    if inplace:
        image.data = nim.data
        nim = image
    else:
        nim = check_framedata(nim)

    return nim


def subtract_bias(image, master_bias, inplace=False):
    """
    Subtract a master_bias frame from a FrameData.

    Notes
    -----
    * This function will just subtract a master bias image from the original
      image. The master bias calculation, by handling several bias images,
      must be done previously.

    * If ``image`` is not a `~astropop.framedata.FrameData` instance,
      inplies in ``inplace=False``, and a new `~astropop.framedata.FrameData`
      instance will be created.

    Parameters
    ----------
    image: `~astropop.framedata.FrameData` compatible
        Image to perform the bias correction. `~astropy.units.Quantity`,
        numerical values and `~astropy.nddata.CCDData` are also suported.
    master_bias: `~astropop.framedata.FrameData` compatible
        Master bias image to be subtracted from the ``image``.
    inplace: bool, optional
        If True, the operations will be performed inplace in the ``image``.

    Returns
    -------
    `~astropop.framedata.FrameData`:
        New bias corrrected `FrameData` instance if ``inplace``, else the
        ``image`` `~astropop.framedata.FrameData` instance.
    """
    master_bias = check_framedata(master_bias)
    nim = imarith(image, master_bias, '-', inplace=False)

    nim.header['hierarch astropop bias_corrected'] = True
    name = master_bias.origin_filename
    if name is not None:
        nim.header['hierarch astropop bias_corrected_file'] = name

    if inplace:
        image.data = nim.data
        nim = image
    else:
        nim = check_framedata(nim)

    return nim


def subtract_dark(image, master_dark, dark_exposure, image_exposure,
                  inplace=False):
    """
    Subtract master_dark frame from a FrameData.

    Notes
    -----
    * This function will just subtract a master dark frame from the original
      image. The master dark calculation, by handling several dark images,
      must be done previously.

    * Different exposure times among dark image and `FrameData` are handled by
      a multiplying the Master dark image by ``image_exposure/dark_exposure``

    * If ``image`` is not a `~astropop.framedata.FrameData` instance,
      inplies in ``inplace=False``, and a new `~astropop.framedata.FrameData`
      instance will be created.

    Parameters
    ----------
    image : `~astropop.framedata.FrameData` compatible
        Image to perform the dark correction. `~astropy.units.Quantity`,
        numerical values and `~astropy.nddata.CCDData` are also suported.
    master_dark : `~astropop.framedata.FrameData` compatible
        Master dark image to be subtracted from the ``image``.
    dark_exposure : float
        Exposure time of the Master dark.
    image_exposure : float
        Exposure time of the `image`.
    inplace : bool, optional
        If True, the operations will be performed inplace in the `image`.

    Returns
    -------
    `~astropop.framedata.FrameData`:
        New dark corrrected `FrameData` instance if ``inplace``, else the
        ``image`` `~astropop.framedata.FrameData` instance.
    """
    image = check_framedata(image)
    master_dark = check_framedata(master_dark)
    scale = image_exposure/dark_exposure
    if scale != 1:
        logger.debug('Scaling dark by %s factor to match image'
                     ' exposure.', scale)
        master_dark = imarith(master_dark, scale, "*", inplace=False)

    nim = imarith(image, master_dark, '-', inplace=inplace)

    nim.header['hierarch astropop dark_corrected'] = True
    nim.header['hierarch astropop dark_corrected_scale'] = scale
    name = master_dark.origin_filename
    if name is not None:
        name = os.path.basename(name)
        nim.header['hierarch astropop dark_corrected_file'] = name

    return nim


def flat_correct(image, master_flat, min_value=None, norm_value=None,
                 inplace=False):
    """
    Divide the image by a flat field frame.

    Parameters
    ----------
    image : `~astropop.framedata.FrameData` compatible
        Image to perform the flat field correction. `~astropy.units.Quantity`,
        numerical values and `~astropy.nddata.CCDData` are also suported.
    master_flat : `~astropop.framedata.FrameData` compatible
        Master flat field image to be subtracted from the ``image``.
    min_value : float, optional
    norm_value : float, optional
    inplace : bool, optional
        If True, the operations will be performed inplace in the `image`.

    Returns
    -------
    `~astropop.framedata.FrameData`:
        New flat field corrected `FrameData` instance if ``inplace``, else the
        ``image`` `~astropop.framedata.FrameData` instance.
    """
    master_flat = check_framedata(master_flat)
    image = check_framedata(image)

    if min_value is not None:
        # FIXME: this will modify master flat!
        logger.debug('Set lower flat value to %s', min_value)
        mask = master_flat.data < min_value
        master_flat.data[np.where(mask)] = min_value

    if norm_value is not None:
        logger.debug('Normalizing flat with %s value.', norm_value)
        master_flat = imarith(master_flat, norm_value, '/', inplace=False)

    nim = imarith(image, master_flat, '/', inplace=inplace)

    nim.header['hierarch astropop flat_corrected'] = True

    name = master_flat.origin_filename
    if name is not None:
        name = os.path.basename(name)
        nim.header['hierarch astropop flat_corrected_file'] = name

    return nim


def process_image(framedata, master_bias=None, master_dark=None,
                  master_flat=None, gain=None, image_exposure=None,
                  dark_exposure=None, trim=None,
                  lacosmic=False, rebin_func=np.sum,
                  rebin_size=None, readnoise=None, badpixmask=None,
                  overscan=None):
    """Process all the default steps of CCD calibration."""
    raise NotImplementedError
