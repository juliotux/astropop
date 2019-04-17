# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..core import Instrument


class SingleCCDCamera(Instrument):
    """Base implementation to handle single ccd câmeras (single HDU images)."""
    _base_image_hdu = 0
    _badpixmask_ext = "badpixmask"
    _identifier = "dummy_singleccdcamera"

    def __init__(self, *args, **kwargs):
        super(SingleCCDCamera, self).__init__(*args, **kwargs)

    def read_raw_file(self, filename):
        """Read a telescope colected file (no badpixmask)."""
        raise NotImplementedError()

    def read_processed_file(self, filename):
        """Read a processed image file."""
        raise NotImplementedError()

    def save_image(self, filename, image):
        """Save one image to filename."""
        raise NotImplementedError()

    def get_image_rect(self, image):
        """Get the rect (b, t, l, r) of the image."""
        raise NotImplementedError()

    def trim_image(self, image, bottom, top, left, right):
        """Trim the image to the rect."""
        raise NotImplementedError()

    def get_bin_size(self, image):
        """Get the bin size of an image."""
        raise NotImplementedError()

    def bin_image(self, image, bin_size):
        """Bin the image to a new bin_size."""
        raise NotImplementedError()

    def get_gain(self, image):
        """Get the gain value of an image."""
        raise NotImplementedError()

    def get_readnoise(self, image):
        """Get the read noise of an image."""
        raise NotImplementedError()

    def get_platescale(self, image):
        """Get the plate scale of an image."""
        raise NotImplementedError()

    def get_filter(self, image):
        """Get the filter name of an image."""
        raise NotImplementedError()
