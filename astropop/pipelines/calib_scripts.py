# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import six
import numpy as np
from astropy.io import fits

from ..image_processing.ccd_processing import process_image
from ..image_processing.register import hdu_shift_images
from ..image_processing.imarith import imcombine as combine
from ..py_utils import process_list, mkdir_p
from ..fits_utils import check_hdu

_mem_limit = 1e9


def _calib_image(image, product_dir=None, master_bias=None, master_flat=None,
                 dark_frame=None, badpixmask=None, prebin=None, gain=None,
                 gain_key='GAIN', rdnoise_key='RDNOISE', exposure_key=None,
                 calib_dir=None, remove_cosmics=True,
                 bias_check_keys=[], flat_check_keys=[],
                 dark_check_keys=[]):
    """Calib one single science image with calibration frames."""
    image = check_hdu(image)

    kwargs = {'rebin_size': prebin,
              'gain_key': gain_key,
              'gain': gain,
              'readnoise_key': rdnoise_key,
              'exposure_key': exposure_key,
              'lacosmic': remove_cosmics,
              'bias_check_keys': bias_check_keys,
              'flat_check_keys': flat_check_keys,
              'dark_check_keys': dark_check_keys}

    if master_bias:
        master_bias = os.path.join(calib_dir, master_bias)
        kwargs['master_bias'] = master_bias
    if master_flat:
        master_flat = os.path.join(calib_dir, master_flat)
        kwargs['master_flat'] = master_flat
    if dark_frame:
        dark_frame = os.path.join(calib_dir, dark_frame)
        kwargs['dark_frame'] = dark_frame
    if badpixmask:
        badpixmask = os.path.join(calib_dir, badpixmask)

    image = process_image(image, **kwargs)

    return image


def create_calib(sources, result_file=None, calib_type=None, master_bias=None,
                 master_flat=None, dark_frame=None, badpixmask=None,
                 prebin=None, gain_key=None, rdnoise_key=None, gain=None,
                 combine_method='median', combine_sigma=None,
                 combine_align_method=None, remove_cosmics=False,
                 exposure_key=None, mem_limit=_mem_limit, calib_dir=None,
                 bias_check_keys=[], flat_check_keys=[],
                 dark_check_keys=[]):
    """Create calibration frames."""
    s = process_list(_calib_image, sources, master_bias=master_bias,
                     master_flat=master_flat, dark_frame=dark_frame,
                     badpixmask=badpixmask, prebin=prebin, gain_key=gain_key,
                     rdnoise_key=rdnoise_key, gain=gain,
                     exposure_key=exposure_key, remove_cosmics=remove_cosmics,
                     calib_dir=calib_dir,
                     bias_check_keys=bias_check_keys,
                     flat_check_keys=flat_check_keys,
                     dark_check_keys=dark_check_keys)

    if combine_sigma is not None:
        reject = 'sigmaclip'
    else:
        reject = []

    if calib_type == 'flat':
        def scaling_func(arr):
            return 1/np.ma.average(arr)
    else:
        scaling_func = None

    if result_file is not None and calib_dir is not None:
        result_file = os.path.join(calib_dir, result_file)

    if combine_align_method in ['fft', 'wcs']:
        s = hdu_shift_images(s, combine_align_method)

    res = combine(s, output_file=result_file, method=combine_method,
                  reject=reject, sigma_clip_low=combine_sigma,
                  sigma_clip_high=combine_sigma, mem_limit=mem_limit,
                  scale=scaling_func)

    if badpixmask is not None and calib_type == 'flat':
        badpix = np.logical_or(res.data <= 0.2, res.data >= 10).astype('uint8')
        badpix = check_hdu(badpix)
        if calib_dir is not None:
            badpixmask = os.path.join(calib_dir, badpixmask)
        badpix.writeto(badpixmask)

    return res


def calib_science(sources, master_bias=None, master_flat=None, dark_frame=None,
                  badpixmask=None, prebin=None, gain_key='GAIN', gain=None,
                  rdnoise_key='RDNOISE', combine_method=None,
                  combine_sigma=None, exposure_key=None, mem_limit=_mem_limit,
                  product_dir=None, combine_align_method=None, calib_dir=None,
                  save_calib_path=None, remove_cosmics=True,
                  return_filename=False, bias_check_keys=[],
                  flat_check_keys=[], dark_check_keys=[]):
    """Calib science images with gain, flat, bias and binning."""
    s = process_list(_calib_image, sources, master_bias=master_bias,
                     master_flat=master_flat, dark_frame=dark_frame,
                     badpixmask=badpixmask, prebin=prebin, gain=gain,
                     rdnoise_key=rdnoise_key, exposure_key=exposure_key,
                     calib_dir=calib_dir, remove_cosmics=remove_cosmics,
                     bias_check_keys=bias_check_keys,
                     flat_check_keys=flat_check_keys,
                     dark_check_keys=dark_check_keys)

    s = process_list(check_hdu, s)

    if combine_align_method in ['fft', 'wcs']:
        s = hdu_shift_images(s, method=combine_align_method)

    if save_calib_path is not None and isinstance(sources[0],
                                                  six.string_types):
        mkdir_p(save_calib_path)
        names = process_list(lambda x: os.path.join(save_calib_path,
                                                    os.path.basename(x)),
                             sources)
        process_list(lambda x: fits.writeto(*x, overwrite=True),
                     zip(names, [i.data for i in s], [i.header for i in s]))
        files_saved = True
    else:
        files_saved = False

    if combine_method is not None:
        if combine_align_method is not None:
            s = hdu_shift_images(s, combine_align_method)
        if combine_sigma is not None:
            reject = ['sigmaclip']
        else:
            reject = []
        s = combine(s, method=combine_method, reject=reject,
                    sigma_clip_low=combine_sigma,
                    sigma_clip_high=combine_sigma,
                    mem_limit=mem_limit)

    if return_filename and files_saved:
        return names

    return s
