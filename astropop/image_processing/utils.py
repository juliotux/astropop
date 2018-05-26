"""Utilities to easy calibrate images and create calibration frames."""

import copy
import numpy as np

from .ccd_processing import process_image
from .imarith import imcombine
from ..py_utils import process_list
from ..fits_utils import fits_yielder


def combine_bias(image_list, save_file=None, rebin_size=None, rebin_func=np.sum,
                 gain=None, gain_key='GAIN', readnoise_key='RDNOISE',
                 combine_method='median', combine_sigma_clip=3,
                 remove_cosmics=False, lacosmic_params={},
                 mem_limit=1e8, **kwargs):
    """Process and combine bias images."""
    hdus = process_list(process_image, image_list, rebin_size=rebin_size,
                        gain=gain, gain_key=gain_key, rebin_func=rebin_func,
                        readnoise_key=readnoise_key, lacosmic=remove_cosmics,
                        lacosmic_params=lacosmic_params)
    hdus = imcombine(hdus, save_file, method=combine_method,
                     reject='sigmaclip', sigma_clip_low=combine_sigma_clip,
                     sigma_clip_high=combine_sigma_clip, mem_limit=mem_limit,
                     overwrite=True)

    return hdus


def combine_dark(image_list, save_file=None, master_bias=None, rebin_size=None,
                 gain=None, gain_key='GAIN', readnoise_key='RDNOISE',
                 combine_method='median', combine_sigma_clip=3,
                 remove_cosmics=False, lacosmic_params={},
                 exposure_key='EXPTIME', rebin_func=np.sum,
                 mem_limit=1e8, **kwargs):
    """Process and combine dark frames."""
    # due to the manipulations needing, no generator allowed
    image_list = list(image_list)
    exps = [i[exposure_key] for i in fits_yielder('header', image_list)]
    for i in range(len(exps)):
        if exps[i] != exps[i-1]:
            raise ValueError('Combine dark frames with different exposures '
                             'not supported.')

    hdus = process_list(process_image, image_list, rebin_size=rebin_size,
                        gain=gain, gain_key=gain_key, rebin_func=rebin_func,
                        readnoise_key=readnoise_key, lacosmic=remove_cosmics,
                        master_bias=master_bias,
                        lacosmic_params=lacosmic_params)

    hdus = imcombine(hdus, save_file, method=combine_method,
                     reject='sigmaclip', sigma_clip_low=combine_sigma_clip,
                     sigma_clip_high=combine_sigma_clip, mem_limit=mem_limit,
                     overwrite=True)

    return hdus


def combine_flat(image_list, save_file=None, master_bias=None, dark_frame=None,
                 rebin_size=None, gain=None, gain_key='GAIN', rebin_func=np.sum,
                 readnoise_key='RDNOISE', combine_method='median',
                 combine_sigma_clip=3, exposure_key='EXPTIME',
                 remove_cosmics=False, lacosmic_params={}, mem_limit=1e8,
                 **kwargs):
    """Process and combine flat images (normalizing)."""
    hdus = process_list(process_image, image_list, rebin_size=rebin_size,
                        master_bias=master_bias, dark_frame=dark_frame,
                        gain=gain, gain_key=gain_key, rebin_func=rebin_func,
                        readnoise_key=readnoise_key, exposure_key=exposure_key,
                        lacosmic=remove_cosmics,
                        lacosmic_params=lacosmic_params)

    def scaling_func(arr):
        return 1/np.nanmean(arr)

    hdus = imcombine(hdus, save_file, method=combine_method,
                     reject='sigmaclip', sigma_clip_low=combine_sigma_clip,
                     sigma_clip_high=combine_sigma_clip, mem_limit=mem_limit,
                     overwrite=True, scale=scaling_func)

    return hdus
