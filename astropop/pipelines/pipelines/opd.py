

from ..pipeline import Stage, Instrument, Processor
from ..common_instruments import SimpleCCDCamera
from ...logger import logger


class OPDAcqCams(SimpleCCDCamera):
    def platescale(self, product=None):
        raise NotImplementedError()

    def fitsrect(self, product=None):
        raise NotImplementedError()

    def binning(self, product=None):
        raise NotImplementedError()

    def gain(self, product=None):
        raise NotImplementedError()

    def readnoise(self, product=None):
        raise NotImplementedError()

    def exposure(self, product=None):
        raise NotImplementedError()

    def tune_calib_frame(self, product):
        """Reprocess things in calib frames in order to be usable.

        like trimming, binning, etc.
        """
        raise NotImplementedError()

    def site(self, product=None):
        """Get the site location, based or not in a filegroup."""
        raise NotImplementedError()

    def timezone(self, product=None):
        """Get the time zone of the observing site."""
        raise NotImplementedError()

    def night(self, product=None):
        raise NotImplementedError()
