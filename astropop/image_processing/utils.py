"""Utilities to easy calibrate images and create calibration frames."""

import numpy as np

from .ccd_processing import process_image
from .imarith import imcombine
from ..py_utils import process_list
from ..fits_utils import fits_yielder


def combine_bias(image_list, save_file=None, save_compress=False,
                 rebin_size=None, rebin_func=np.sum,
                 gain=None, gain_key='GAIN', readnoise_key='RDNOISE',
                 combine_method='median', combine_sigma_clip=3,
                 remove_cosmics=False, lacosmic_params={},
                 mem_limit=1e8, inplace=True, add_keywords={}, **kwargs):
    """Process and combine bias images."""
    hdus = process_list(process_image, image_list, rebin_size=rebin_size,
                        gain=gain, gain_key=gain_key, rebin_func=rebin_func,
                        readnoise_key=readnoise_key, lacosmic=remove_cosmics,
                        lacosmic_params=lacosmic_params, inplace=inplace)

    for i, v in add_keywords.items():
        for h in hdus:
            h.header[i] = v

    hdus = imcombine(hdus, save_file, method=combine_method,
                     reject='sigmaclip', sigma_clip_low=combine_sigma_clip,
                     sigma_clip_high=combine_sigma_clip, mem_limit=mem_limit,
                     overwrite=True, save_compress=save_compress)

    return hdus


def combine_dark(image_list, save_file=None, save_compress=False,
                 master_bias=None, rebin_size=None,
                 gain=None, gain_key='GAIN', readnoise_key='RDNOISE',
                 combine_method='median', combine_sigma_clip=3,
                 remove_cosmics=False, lacosmic_params={},
                 exposure_key='EXPTIME', rebin_func=np.sum,
                 mem_limit=1e8, inplace=True, add_keywords={}, **kwargs):
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
                        master_bias=master_bias, inplace=inplace,
                        lacosmic_params=lacosmic_params)

    for i, v in add_keywords.items():
        for h in hdus:
            h.header[i] = v

    hdus = imcombine(hdus, save_file, method=combine_method,
                     reject='sigmaclip', sigma_clip_low=combine_sigma_clip,
                     sigma_clip_high=combine_sigma_clip, mem_limit=mem_limit,
                     overwrite=True, save_compress=save_compress)

    return hdus


def combine_flat(image_list, save_file=None, save_compress=False,
                 master_bias=None, dark_frame=None,
                 rebin_size=None, gain=None, gain_key='GAIN', rebin_func=np.sum,
                 readnoise_key='RDNOISE', combine_method='median',
                 combine_sigma_clip=3, exposure_key='EXPTIME',
                 remove_cosmics=False, lacosmic_params={}, mem_limit=1e8,
                 inplace=True, add_keywords={}, **kwargs):
    """Process and combine flat images (normalizing)."""
    hdus = process_list(process_image, image_list, rebin_size=rebin_size,
                        master_bias=master_bias, dark_frame=dark_frame,
                        gain=gain, gain_key=gain_key, rebin_func=rebin_func,
                        readnoise_key=readnoise_key, exposure_key=exposure_key,
                        lacosmic=remove_cosmics, inplace=inplace,
                        lacosmic_params=lacosmic_params)

    for i, v in add_keywords.items():
        for h in hdus:
            h.header[i] = v

    def scaling_func(arr):
        return 1/np.nanmean(arr)

    hdus = imcombine(hdus, save_file, method=combine_method,
                     reject='sigmaclip', sigma_clip_low=combine_sigma_clip,
                     sigma_clip_high=combine_sigma_clip, mem_limit=mem_limit,
                     overwrite=True, scale=scaling_func,
                     save_compress=save_compress)

    return hdus
