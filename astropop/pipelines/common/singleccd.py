# Licensed under a 3-clause BSD style license - see LICENSE.rst

import abc

from astropy.io.fits import Header

from ..core import Instrument
from ...fits_utils import check_ccddata, check_header_keys


__all__ = ["SingleCCDCamera"]


class IncompatibleCalibError(Exception):
    """Error raised when the calib and science frame are incompatible."""


class SingleCCDCamera(Instrument):
    """Base implementation to handle single ccd câmeras (single HDU images)."""
    _base_image_hdu = 0
    _raw_unit = 'adu'
    _identifier = "dummy_singleccdcamera"
    _protected_keywords = []
    _base_raw_keyword = 'hierarch raw_data {}'
    _bias_check_keys = []
    _flat_check_keys = []
    _dark_check_keys = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_raw_file(self, filename):
        """Read a telescope colected file (no badpixmask)."""
        ccd = check_ccddata(filename, ext=self._base_image_hdu,
                            utype=self._raw_unit)

        header = Header()
        for key, value in ccd.meta.items():
            if key in self._protected_keywords:
                header[key] = value
            else:
                header[self._base_raw_keyword.format(key)] = value
        ccd.meta = header
        return ccd

    def save_fits(self, ccddata, filename):
        """Save a CCDData (w badpix and uncert) to a fits file."""
        ccddata.write(filename)

    def read_fits(self, filename):
        """Read a fits file to a CCDData, recommended for processed files."""
        return check_ccddata(filename)

    def check_compatibility(self, ccddata, calib):
        """Check the compatibility between a CCDData and a calib frame.

        Returns:
            calib : CCDData
                The calib frame prepared to correct the science frame.
                (with block_reduce, cutout, etc.)

        Raise:
            IncompatibleCalibError:
                Raised when the ccddata and calib frame are not compatible.
        """
        calib = self.read_fits(calib)

        if not hasattr(calib, 'calib_type'):
            raise ValueError('Calib frame {} is invalid'.format(calib))
        calib_type = calib.type

        if hasattr(ccddata, 'calib_steps'):
            for i in ccddata.calib_steps:
                if i not in calib.calib_steps:
                    raise IncompatibleCalibError("Calib ")

        if self.get_bin_size(ccddata) != self.get_bin_size(calib):
            # different binnings means incompatible
            raise IncompatibleCalibError('ccddata and calib have different'
                                         ' binnings: {} {}'
                                         .format(self.get_bin_size(ccddata),
                                                 self.get_bin_size(calib)))

        if self.get_image_rect(ccddata) != self.get_image_rect(calib):
            # different rects can be fixed
            calib_rect = self.get_image_rect(calib)
            ccd_rect = self.get_image_rect(ccddata)
            self.logger.debug('Trimming calib from {} to {}'
                              .format(calib_rect, ccd_rect))
            calib = self.trim_image(calib, ccd_rect)

        if ccddata.shape != calib.shape:
            # different final shapes means incompatible
            raise IncompatibleCalibError('ccddata and calib have different'
                                         ' shapes: {} {}'
                                         .format(ccddata.shape,
                                                 calib.shape))

        if calib_type == 'flat':
            if self.get_filter(ccddata) != self.get_filter(calib):
                raise IncompatibleCalibError('ccddata and calib have different'
                                             ' filters: {} {}'
                                             .format(self.get_filter(ccddata),
                                                     self.get_filter(calib)))

        check_keys = []
        if calib_type == 'bias':
            check_keys = [self._base_raw_keywords.format(i)
                          for i in self._bias_check_keys]
        elif calib_type == 'dark':
            check_keys = [self._base_raw_keywords.format(i)
                          for i in self._dark_check_keys]
        elif calib_type == 'flat':
            check_keys = [self._base_raw_keywords.format(i)
                          for i in self._flat_check_keys]
        check_header_keys(ccddata, calib, keywords=check_keys,
                          logger=self.logger)

        return calib

    @abc.abstractmethod
    def get_image_rect(self, ccddata):
        """Get the rect (b, t, l, r) of the image."""
        raise NotImplementedError()

    @abc.abstractmethod
    def trim_image(self, ccddata, rect):
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
    def get_exposure(self, ccddata):
        """Get the exposure of an image."""
        raise NotImplementedError()
