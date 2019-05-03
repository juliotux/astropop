# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import numpy as np
import ccdproc

from ..logger import logger
from .imarith import imarith
from ..fits_utils import check_ccddata


__all__ = ['cosmics_lacosmic']

# TODO: subtract_dark
# TODO: subtract_overscan
# TODO: divide_flat
# TODO: block_reduce
# TODO: block_replicate
# TODO: trim_image


# def trim_image(image, section, fits_convention=False, inplace=False,
#                logger=logger):
#     """Trim a section from a image

#     If original_section passed, it will be used as reference for trimming.
#     """
#     if not inplace:
#         im = copy.copy(image)
#     else:
#         im = image

#     slices = slices_from_string(section, fits_convention=fits_convention)

#     # check boundaries
#     shape = im.data.shape
#     for i, m in zip(slices, shape):
#         m = int(m)
#         l = [i.start or 0, i.stop or m]
#         if np.min(l) < 0 or i.stop > m:
#             raise ValueError('Slice out of the limits of the image.')

#     ndata = np.array(im.data[slices])
#     im.data = ndata
#     im.header['trimmed'] = True
#     im.header['trimmed slice'] = section
#     im.header['trimmed fits_convention'] = fits_convention
#     return im


# def subtract_dark(image, dark_frame, dark_exposure=None, image_exposure=None,
#                   exposure_key=None, check_keys=[], inplace=False,
#                   logger=logger):
#     """Subtract dark frame from an image, scaling by exposure."""
#     if len(check_keys) > 0:
#         check_header_keys(image, dark_frame, check_keys)
#     dark = check_hdu(dark_frame)
#     imag = check_hdu(image)

#     if exposure_key is not None:
#         try:
#             d_exp = opd_number(dark.header[exposure_key])
#             i_exp = opd_number(imag.header[exposure_key])
#             if dark_exposure is not None or image_exposure is not None:
#                 logger.info('exposure_key and data exposures passed to the '
#                             'function. Using only exposure_key')
#         except KeyError:
#             logger.warn('exposure_key is invalid, using dark_exposure and '
#                         'image_exposure instead.')
#             d_exp = dark_exposure
#             i_exp = image_exposure

#     if d_exp is not None and i_exp is not None:
#         logger.debug('{} {}'.format(i_exp, d_exp))
#         scale = i_exp/d_exp
#         dark = imarith(dark, scale, '*', inplace=False)
#         logger.debug('dark_frame values scaled by {} to fit the image'
#                      ' exposure.'.format(scale))
#     else:
#         scale = None
#         logger.debug('No exposures for image and/or dark_frame found.'
#                      ' Ignoring scaling.')

#     nim = imarith(imag, dark, '-', inplace=inplace)
#     nim.header['hierarch dark corrected'] = True
#     darkfile = dark.fileinfo()
#     if darkfile is not None:
#         name = darkfile['file'].name
#         nim.header['hierarch dark frame'] = os.path.basename(name)
#     if scale is not None:
#         nim.header['hierarch dark scaled'] = scale

#     return nim


# def divide_flat(image, master_flat, check_keys=[], inplace=False,
#                 logger=logger):
#     """Divide a image by a master flat field frame."""
#     if len(check_keys) > 0:
#         check_header_keys(image, master_flat, check_keys)
#     nim = imarith(image, master_flat, '-', inplace=inplace)

#     nim.header['hierarch flat corrected'] = True
#     flatfile = check_hdu(master_flat).fileinfo()
#     if flatfile is not None:
#         name = flatfile['file'].name
#         nim.header['hierarch flat master'] = os.path.basename(name)

#     return nim


# def block_reduce(image, block_size, func=np.sum, readnoise_key=None,
#                  inplace=False, logger=logger):
#     """Process rebinnig in one image. Like block_reduce."""
#     im = check_hdu(image)
#     if not inplace:
#         im = fits.PrimaryHDU(im.data, header=im.header)

#     im.data = br(im.data, block_size, func)
#     summed = br(im.data, block_size, np.sum)
#     im.header['hierarch block_reduced'] = block_size

#     # Fix readnoise if readnoise_key is passed
#     if readnoise_key in im.header.keys():
#         try:
#             rdnoise = opd_number(im.header[readnoise_key])
#             im.header[readnoise_key] = block_size * rdnoise
#             if func == np.sum:
#                 # fix the readnoise by functions that are not the sum
#                 im.header[readnoise_key] /= np.sqrt(np.nanmean(summed/
#                                                                im.data))
#             im.header['hierarch block_reduced readnoise_old'] = rdnoise
#         except ValueError:
#             pass

#     return im


# block_reduce.__doc__ += br.__doc__


def cosmics_lacosmic(ccddata, logger=logger, **lacosmic_kwargs):
    """Remove cosmic rays with LAcosmic. From astroscrappy package."""
    # As lacosmic removes and replace the cosmics pixels, no need to update
    # the mask
    nccd = ccdproc.cosmicray_lacosmic(ccddata, **lacosmic_kwargs)
    ccddata.data = nccd.data
    ccddata.mask = nccd.mask
    ccddata.header['hierarch astropop lacosmic'] = True
    return ccddata


def gain_correct(image, gain=None, inplace=False,
                 logger=logger):
    """Process the gain correction of an image."""
    # TODO: wrap ccdproc?
    image = check_ccddata(image)
    nim = imarith(image, gain, '*', inplace=inplace)
    nim.header['hierarch astropop gain_corrected'] = True
    nim.header['hierarch astropop gain_corrected_value'] = gain
    return nim


def subtract_bias(image, master_bias, inplace=False,
                  logger=logger):
    """Subtract a master_bias frame from an image."""
    # TODO: wrap ccdproc?
    master_bias = check_ccddata(master_bias)
    nim = imarith(image, master_bias, '-', inplace=inplace)

    nim.header['hierarch astropop bias_corrected'] = True
    biasfile = master_bias.filename
    if biasfile is not None:
        name = biasfile
        nim.header['hierarch astropop master_bias'] = os.path.basename(name)

    return nim


def process_ccd(ccddata, master_bias=None, master_dark=None, master_flat=None,
                gain=None, image_exposure=None, dark_exposure=None, trim=None,
                lacosmic=False, lacosmic_params={}, rebin_func=np.sum,
                rebin_size=None, readnoise=None, badpixmask=None,
                logger=logger):
    """Process all the default steps of CCD calibration."""
    # TODO: implement here

    return ccddata
