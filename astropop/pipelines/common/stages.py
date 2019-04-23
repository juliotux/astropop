# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..core import Stage


__all__ = ['ImageProcessing', 'SourceExtraction', 'AperturePhotometry',
           'PSFPhotometry', 'AstrometryCalib', 'SavePhotometryFITS',
           'SavePhotometryASDF', 'SaveImageFITS', 'SaveImageASDF',
           'MultiImageProcessing', 'PhotometryCalib']


class ImageProcessing(Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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

# TODO: Continue de design
