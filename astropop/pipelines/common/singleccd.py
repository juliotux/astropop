# Licensed under a 3-clause BSD style license - see LICENSE.rst

import abc

from ..core import Instrument


__all__ = ["SingleCCDCamera"]


class SingleCCDCamera(Instrument):
    """Base implementation to handle single ccd câmeras (single HDU images)."""
    _base_image_hdu = 0
    _badpixmask_ext = "badpixmask"
    _identifier = "dummy_singleccdcamera"

    def __init__(self, *args, **kwargs):
        super(SingleCCDCamera, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def read_raw_file(self, filename):
        """Read a telescope colected file (no badpixmask)."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_image_rect(self, ccddata):
        """Get the rect (b, t, l, r) of the image."""
        raise NotImplementedError()

    @abc.abstractmethod
    def trim_image(self, ccddata, bottom, top, left, right):
        """Trim the image to the rect."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_bin_size(self, ccddata):
        """Get the bin size of an image."""
        raise NotImplementedError()

    @abc.abstractmethod
    def bin_image(self, ccddata, bin_size):
        """Bin the image to a new bin_size."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_gain(self, ccddata):
        """Get the gain value of an image."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_readnoise(self, ccddata):
        """Get the read noise of an image."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_platescale(self, ccddata):
        """Get the plate scale of an image."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_filter(self, ccddata):
        """Get the filter name of an image."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_unit(self, ccddata):
        """Get the unit of the image data."""
