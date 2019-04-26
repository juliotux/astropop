# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..core import Stage

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
