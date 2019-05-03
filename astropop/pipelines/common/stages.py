# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from ..core import Stage
from ...image_processing.ccd_processing import cosmic_lacosmic

# TODO: Continue de design

__all__ = ['ImageProcessing', 'SourceExtraction', 'AperturePhotometry',
           'PSFPhotometry', 'AstrometryCalib', 'SavePhotometryFITS',
           'SavePhotometryASDF', 'SaveImageFITS', 'SaveImageASDF',
           'MultiImageProcessing', 'PhotometryCalib']


class ImageProcessing(Stage):
    """Basic image processing for single CCD products.

    Product Needs:
        Capabilities:
            raw_filenames = list of raw science files to process
            calib_filenames = list of calibration frames filenames
        Instrument:
            All singleccd instrument functions.
    """
    _requested_capabilities = ['raw_filenames', 'calib_filenames']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, product, config):
        product.add_capability('calibed_ccddata')
        product.calibed_ccddata = []
        instrument = product.instrument
        logger = product.logger

        cache_folder = config.get('cache_folder',
                                  os.path.expanduser('~/astropop_cache'))

        for i in product.raw_filenames:
            # Read and cache the files
            ccd = instrument.read_raw_file(i)
            name = os.path.basename(i)
            ccd.enable_memmap(cache_folder=cache_folder, filename=name)

            lacosmic = config.get('lacosmic', True)
            if lacosmic:
                lacosmic_kwargs = config.get('lacosmic_args', {})
                cosmic_lacosmic(ccd, logger=logger, **lacosmic_kwargs)

            # Now, do the calibrations
            if 'bias' in product.calib_filenames:
                print('do bias')

            if 'dark' in product.calib_filenames:
                print('do dark')

            if 'flat' in product.calib_filenames:
                print('do flat')

            if 'badpix' in product.calib_filenames:
                print('apply badpix mask')
            # TODO: Continue here
            product.calibed_ccddata.append(ccd)
        return


class SourceExtraction(Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AperturePhotometry(Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PSFPhotometry(Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AstrometryCalib(Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PhotometryCalib(Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PolarimetryCompute(Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SavePhotometryFITS(Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SavePhotometryASDF(Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SaveImageFITS(Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SaveImageASDF(Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MultiImageProcessing(Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
