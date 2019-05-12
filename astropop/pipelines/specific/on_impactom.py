# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.coordinates import SkyCoord
from astropy import units as u

from ...image_processing import ccdproc
from ..common.singleccd import SingleCCDCamera


class ImpactonCamera(SingleCCDCamera):
    """Provides the interface to Impacton camera."""
    _base_image_hdu = 0
    _raw_unit = 'adu'
    _identifier = "impacton_apogee"
    _protected_keywords = ['observer', 'object', 'imagetyp',
                           'simple', 'bitpix', 'naxis', 'naxis1',
                           'naxis2', 'bscale', 'bzero', 'date-obs',
                           'exptime', 'exposure', 'night']
    _bias_check_keys = ['xbinning', 'ybinnig', 'night',
                        'telescop', 'instrume', 'swcreate']
    _flat_check_keys = ['xbinning', 'ybinnig', 'night',
                        'telescop', 'instrume', 'swcreate',
                        'filter']
    _dark_check_keys = ['xbinning', 'ybinnig', 'night',
                        'telescop', 'instrume', 'swcreate']

    def read_raw_file(self, filename, **kwargs):
        ccd = super().read_raw_file(filename, **kwargs)

        coords = self.get_coordinates(ccd).to_string('hmsdms').split(' ')
        ccd.meta['RA'] = (coords[0],
                          'Nominal Right Ascension coordinate in HH MM SS')
        ccd.meta['DEC'] = (coords[1],
                           'Nominal Declination coordinate in +DD MM SS')

        return ccd

    def get_coordinates(self, ccddata):
        """Read the pointing coordinates of the image."""
        ra = self._get_keyword('OBJCTRA', ccddata.meta)
        dec = self._get_keyword('OBJCTDEC', ccddata.meta)
        return SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')

    def get_image_rect(self, ccddata):
        """Get the rect (b, t, l, r) of the image."""
        self.logger.warn('Image rect not supported for Impacton camera')
        return None

    def trim_image(self, ccddata, rect):
        """Trim the image to the rect."""
        self.logger.warn('Image trim not supported for Impacton camera')
        return ccddata

    def get_bin_size(self, ccddata):
        """Get the bin size of an image (x, y)."""
        xbin = self._get_keyword('XBINNING', ccddata.meta)
        ybin = self._get_keyword('YBINNING', ccddata.meta)

        return (xbin, ybin)

    def bin_image(self, ccddata, bin_size):
        """Bin the image to a new bin_size."""
        return ccdproc.block_reduce(ccddata, bin_size)

    def get_gain(self, ccddata):
        """Get the gain value of an image."""
        # FIXME: check the default values of the instrument
        return 1.0*u.dimensionless_unscaled

    def get_readnoise(self, ccddata):
        """Get the read noise of an image."""
        # FIXME: check the default values of the instrument
        return None

    def get_platescale(self, ccddata):
        """Get the plate scale of an image."""
        return 

    def get_filter(self, ccddata):
        """Get the filter name of an image."""
        raise NotImplementedError()

    def get_exposure(self, ccddata):
        """Get the exposure of an image."""
        raise NotImplementedError()
