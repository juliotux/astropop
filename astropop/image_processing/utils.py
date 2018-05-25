"""Utilities to easy calibrate images and create calibration frames."""

from .ccd_processing import process_image
from .imarith import imcombine
from ..py_utils import process_list
from ..fits_utils import fits_yielder


def combine_bias(image_list, save_file=None, prebin=None,
                 gain=None, gain_key='GAIN', rdnoise_key='RDNOISE',
                 combine_method='median', combine_sigma_clip=3,
                 remove_cosmics=False, lacosmic_params={},
                 mem_limit=1e8):
    """Process and combine bias images."""
    hdus = process_list(process_image, image_list, prebin=prebin,
                        gain=gain, gain_key=gain_key, rdnoise_key=rdnoise_key,
                        lacosmic=remove_cosmics,
                        lacosmic_params=lacosmic_params)
    hdus = imcombine(hdus, save_file, method=combine_method,
                     reject='sigmaclip', sigma_clip_low=combine_sigma_clip,
                     sigma_clip_high=combine_sigma_clip, mem_limit=mem_limit)

    return hdus

def combine_dark(image_list, save_file=None, master_bias=None, prebin=None,
                 gain=None, gain_key='GAIN', rdnoise_key='RDNOISE',
                 combine_method='median', combine_sigma_clip=3,
                 remove_cosmics=False, lacosmic_params={},
                 exposure_key='EXPTIME',
                 mem_limit=1e8):
    """Process and combine dark frames."""
    exps = [i[exposure_key] for i in fits_yielder('header', image_list)]
    for i in range(len(image_list)):
        if exps[i] != exps[i-1]:
            raise ValueError('Combine dark frames with different exposures '
                             'not supported.')

    hdus = process_list(process_image, image_list, prebin=prebin,
                        gain=gain, gain_key=gain_key, rdnoise_key=rdnoise_key,
                        lacosmic=remove_cosmics, master_bias=master_bias,
                        lacosmic_params=lacosmic_params)

    hdus = imcombine(hdus, save_file, method=combine_method,
                     reject='sigmaclip', sigma_clip_low=combine_sigma_clip,
                     sigma_clip_high=combine_sigma_clip, mem_limit=mem_limit)

    return hdus

def combine_flat(image_list, save_file=None, master_bias=None, master_dark=None,
                 prebin=None, gain=None, gain_key='GAIN', rdnoise_key='RDNOISE',
                 combine_method='median', combine_sigma_clip=3,
                 exposure_key='EXPTIME', remove_cosmics=False,
                 lacosmic_params={}, mem_limit=1e8):
    """Process and combine flat images (normalizing)."""
    hdus = process_list(process_image, image_list, prebin=prebin,
                        master_bias=master_bias, master_dark=master_dark,
                        gain=gain, gain_key=gain_key, rdnoise_key=rdnoise_key,
                        lacosmic=remove_cosmics, exposure_key=exposure_key,
                        lacosmic_params=lacosmic_params)

    hdus = imcombine(hdus, save_file, method=combine_method,
                     reject='sigmaclip', sigma_clip_low=combine_sigma_clip,
                     sigma_clip_high=combine_sigma_clip, mem_limit=mem_limit)

    return hdus
