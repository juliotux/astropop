# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from functools import partial
import numpy as np
import ccdproc

from ..logger import logger
from .imarith import imarith
from ..fits_utils import check_ccddata


__all__ = ['cosmics_lacosmic']


# TODO: replace ccdproc functions by built-in, skiping units


block_reduce = ccdproc.block_reduce
block_replicate = ccdproc.block_replicate
trim_image = ccdproc.trim_image
subtract_overscan = partial(ccdproc.subtract_overscan,
                            add_keyword='hierarch astropop'
                            ' overscan_subtracted')


def cosmics_lacosmic(ccddata, logger=logger, **lacosmic_kwargs):
    """Remove cosmic rays with LAcosmic. From astroscrappy package."""
    # As lacosmic removes and replace the cosmics pixels, no need to
    # update the mask
    nccd = ccdproc.cosmicray_lacosmic(ccddata, **lacosmic_kwargs)
    ccddata.data = nccd.data
    ccddata.mask = nccd.mask
    ccddata.header['hierarch astropop lacosmic'] = True
    return ccddata


def gain_correct(image, gain, gain_unit=None, inplace=False,
                 logger=logger):
    """Process the gain correction of an image."""
    nim = ccdproc.gain_correct(image, gain, gain_unit)
    nim.header['hierarch astropop gain_corrected'] = True
    nim.header['hierarch astropop gain_corrected_value'] = gain
    nim.header['hierarch astropop gain_corrected_unit'] = str(gain_unit)

    if inplace:
        image.data = nim.data
        nim = image
    else:
        nim = check_ccddata(nim)

    return nim


def subtract_bias(image, master_bias, inplace=False,
                  logger=logger):
    """Subtract a master_bias frame from a CCDData."""
    master_bias = check_ccddata(master_bias)
    nim = ccdproc.subtract_bias(image, master_bias)

    nim.header['hierarch astropop bias_corrected'] = True
    name = master_bias.filename
    if name is not None:
        nim.header['hierarch astropop bias_corrected_file'] = name

    if inplace:
        image.data = nim.data
        nim = image
    else:
        nim = check_ccddata(nim)

    return nim


def process_ccd(ccddata, master_bias=None, master_dark=None, master_flat=None,
                gain=None, image_exposure=None, dark_exposure=None, trim=None,
                lacosmic=False, lacosmic_params={}, rebin_func=np.sum,
                rebin_size=None, readnoise=None, badpixmask=None,
                overscan=None,
                logger=logger):
    """Process all the default steps of CCD calibration."""

    # TODO: implement here
    return ccdproc.ccd_process(ccddata, oscan=overscan, trim=trim, error=False,
                               master_bias=master_bias, dark_frame=master_dark,
                               master_flat=master_flat,
                               bad_pixel_mask=badpixmask,
                               gain=gain, readnoise=readnoise,
                               data_exposure=image_exposure,
                               dark_exposure=dark_exposure)


###############################################################################
# DONE
###############################################################################


def subtract_dark(image, master_dark, dark_exposure, image_exposure,
                  inplace=False, logger=logger):
    """Subtract master_dark frame from a CCDData."""
    image = check_ccddata(image)
    master_dark = check_ccddata(master_dark)
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
    master_flat = check_ccddata(master_flat)
    image = check_ccddata(image)

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
