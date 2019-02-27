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
from .pipeline import Stage, Processor, Config

_mem_limit = 1e9


class CCCDPreprocess(Stage):
    config = Config(remove_cosmics=True,
                    save_subfolder=None,
                    align_method=None,
                    save_fmt='.fits.gz')

    def run(self, product, config, instrument):
        raw_fg = product.raw_filegroup
