# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import numpy as np
import astroscrappy

from ..logger import logger
from .imarith import imarith
# from ..framedata import check_framedata


__all__ = ['cosmics_lacosmic']


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


def cosmics_lacosmic(ccddata, inplace=False, logger=logger, **lacosmic_kwargs):
    """Remove cosmic rays with LAcosmic. From astroscrappy package."""
    # As lacosmic removes and replace the cosmics pixels, no need to
    # update the mask
    _, dat = astroscrappy.detect_cosmics(ccddata.data, **lacosmic_kwargs)

    if inplace:
        ccd = ccddata
    else:
        ccd = ccddata.copy()

    ccd.data = dat
    # Do not update mask, astroscrappy replace the pixels
    # ccd.mask &= mask
    ccd.header['hierarch astropop lacosmic'] = True
    return ccd


def gain_correct(image, gain, gain_unit=None, inplace=False,
                 logger=logger):
    """Process the gain correction of an image."""
    # TODO: handle units
    nim = imarith(image, gain, '*', inplace=False, logger=logger)
    nim.header['hierarch astropop gain_corrected'] = True
    nim.header['hierarch astropop gain_corrected_value'] = gain
    nim.header['hierarch astropop gain_corrected_unit'] = str(gain_unit)

    if inplace:
        image.data = nim.data
        nim = image
    else:
        nim = check_framedata(nim)

    return nim


def subtract_bias(image, master_bias, inplace=False,
                  logger=logger):
    """Subtract a master_bias frame from a CCDData."""
    master_bias = check_framedata(master_bias)
    nim = imarith(image, master_bias, '-', inplace=False, logger=logger)

    nim.header['hierarch astropop bias_corrected'] = True
    name = master_bias.filename
    if name is not None:
        nim.header['hierarch astropop bias_corrected_file'] = name

    if inplace:
        image.data = nim.data
        nim = image
    else:
        nim = check_framedata(nim)

    return nim


def subtract_dark(image, master_dark, dark_exposure, image_exposure,
                  inplace=False, logger=logger):
    """Subtract master_dark frame from a CCDData."""
    image = check_framedata(image)
    master_dark = check_framedata(master_dark)
    scale = image_exposure/dark_exposure
    if scale != 1:
        logger.debug('Scaling dark by {} factor to match image exposure.'
                     .format(scale))
        master_dark = imarith(master_dark, scale, "*", inplace=False,
                              logger=logger)

    nim = imarith(image, master_dark, '-', inplace=inplace, logger=logger)

    nim.header['hierarch astropop dark_corrected'] = True
    nim.header['hierarch astropop dark_corrected_scale'] = scale
    name = master_dark.filename
    if name is not None:
        name = os.path.basename(name)
        nim.header['hierarch astropop dark_corrected_file'] = name

    return nim


def flat_correct(image, master_flat, min_value=None, norm_value=None,
                 inplace=False, logger=logger):
    """Divide the image by a flat field frame."""
    master_flat = check_framedata(master_flat)
    image = check_framedata(image)

    if min_value is not None:
        logger.debug('Set lower flat value to {}'.format(min_value))
        mask = master_flat.data < min_value
        master_flat.data[np.where(mask)] = min_value

    if norm_value is not None:
        logger.debug('Normalizing flat with {} value.'.format(norm_value))
        master_flat = imarith(master_flat, norm_value, '/',
                              inplace=False, logger=logger)

    nim = imarith(image, master_flat, '/', inplace=inplace, logger=logger)

    nim.header['hierarch astropop flat_corrected'] = True

    name = master_flat.filename
    if name is not None:
        name = os.path.basename(name)
        nim.header['hierarch astropop flat_corrected_file'] = name

    return nim


def process_ccd(ccddata, master_bias=None, master_dark=None, master_flat=None,
                gain=None, image_exposure=None, dark_exposure=None, trim=None,
                lacosmic=False, rebin_func=np.sum,
                rebin_size=None, readnoise=None, badpixmask=None,
                overscan=None,
                logger=logger):
    """Process all the default steps of CCD calibration."""

    raise NotImplementedError
