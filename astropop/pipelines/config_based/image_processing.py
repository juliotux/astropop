# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from collections import OrderedDict
from ..calib_scripts import create_calib
from .base import ReducePipeline
from ...py_utils import batch_key_replace, mkdir_p
from ...image_processing import (register_available_methods,
                                 combine_available_methods)


class CalibPipeline(ReducePipeline):
    def __init__(self, config=None):
        super(CalibPipeline, self).__init__(config=config)

    @property
    def parameters(self):
        prpd = OrderedDict(
            calib_type="Type of calibration frame to generate: 'bias', 'flat',"
                       " 'dark' or 'science'",
            gain_key="Gain header keyword",
            gain="Manual set the gain of the image",
            rdnoise_key="Read noise header keyword",
            exposure_key="Exposure header keyword",
            exposure_type="Unit of exposure: 'sec', 'min', 'hour'",
            time_key="Exposure start header keyword",
            time_key_type="Exposure start header kwyowrd type: any valid "
                          "astropy.time.Time format.",
            save_calib_path="Path to save individual calibrated images",
            raw_dir="Directory containing the raw data",
            product_dir="Directory to save the product data",
            calib_dir="Directory containing the calibration frames",
            sources="List of source files",
            source_ls_pattern="Pattern to load sources from ls command",
            prebin="Number of pixels to rebin the image",
            plate_scale="Plate scale of the field in arcsec/pixel",
            filter="Filter or band used. Will overwrite FILTER header keyword",
            master_bias="Master bias to be subtracted",
            dark_frame="Dark frame to be scaled and subtracted",
            master_flat="Mater flat to divide the image",
            badpixmask="Bad pixel mask. 0=good, 1=bad. "
                       "If not exists, code will generate from clipping flat",
            align_method="Method to align the images: {}"
                         .format(register_available_methods),
            combine_method="Method to combine images: {}"
                           .format(combine_available_methods),
            remove_cosmics="Remove cosmics with astroscrappy. Bool.",
            mem_limit="Maximum memory to use in image combining",
            bias_check_keys="List of keywords in header to check in bias "
                            "correction",
            flat_check_keys="List of keywords in header to check in flat "
                            "correction",
            dark_check_keys="List of keywords in header to check in dark "
                            "correction",)
        return prpd

    def run(self, name, **config):
        for k in config.keys():
            batch_key_replace(config, k)
        s = [os.path.join(config['raw_dir'], i) for i in config['sources']]

        if 'result_file' not in config.keys():
            outname = name
        else:
            outname = config['result_file']

        check_exist = config.get('check_exist', False)
        outf = os.path.join(config['calib_dir'], outname)
        if check_exist and os.path.isfile(outf):
            return

        calib_kwargs = {}
        for i in ['calib_type', 'master_bias', 'master_flat', 'dark_frame',
                  'badpixmask', 'prebin', 'gain_key', 'rdnoise_key', 'gain',
                  'combine_method', 'combine_sigma', 'exposure_key',
                  'mem_limit', 'calib_dir',
                  'bias_check_keys', 'flat_check_keys', 'dark_check_keys']:
            if i in config.keys():
                calib_kwargs[i] = config[i]

        mkdir_p(config['calib_dir'])
        return create_calib(s, outname, **calib_kwargs)
