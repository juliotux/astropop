from astropy.time import Time, TimezoneInfo
from astropy.coordinates import Angle, EarthLocation
from astropy import units
import numpy as np


from .common_processing import SimpleCalibPipeline
from .common_photometry import StackedPhotometryPipeline
from ...catalogs import default_catalogs
from ...fits_utils import check_hdu
from ...logger import logger
from ...file_manager import FileManager


cat_list = {'B': default_catalogs['APASS'],
            'V': default_catalogs['APASS'],
            'R': default_catalogs['GSC2.3'],
            'I': default_catalogs['DENIS']}


class ImpactonCalib(SimpleCalibPipeline):
    template = 'IMPACTON_calib'
    _science_group_keywords = ['object', 'xbinning', 'ybinning', 'filter']
    _science_select_rules = dict(imagetyp=['LIGHT'],
                                 instrume='Apogee USB/Net',
                                 telescop='AstroOptik 1m')
    _science_name_keywords = ['object', 'night', 'filter', 'xbinning',
                              'ybinning']
    _bias_select_keywords = ['instrume', 'telescop', 'xbinning',
                             'ybinning', 'night']
    _bias_select_rules = dict(imagetyp=['BIAS'])
    _bias_name_keywords = ['night', 'ccdsum', 'gain', 'xbinning', 'ybinning']
    _flat_select_keywords = ['instrume', 'telescop', 'filter', 'xbinning',
                             'ybinning', 'night']
    _flat_select_rules = dict(imagetyp=['FLAT'])
    _flat_name_keywords = ['night', 'ccdsum', 'gain', 'filter' 'xbinning',
                           'ybinning']
    calib_process_params = dict(gain_key=None,
                                gain=None,
                                combine_method='median',
                                combine_sigma_clip=3,
                                remove_cosmics=True,
                                mem_limit=1e9,
                                exposure_key='exptime')
    sci_process_params = dict(gain_key=None,
                              gain=None,
                              lacosmic=True,
                              exposure_key='exptime',
                              inplace=True)
    astrometry_params = dict(ra_key='OBJCTRA',
                             dec_key='OBJCTDEC')
    _save_subfolder = '{object}/{night}'
    _save_fits_ext = 0
    _save_fits_fmt = '.fit'
    _save_fits_compressed = False

    def __init__(self, product_dir=None, calib_dir=None,
                 ext=0, fits_extensions=['.fit'], compression=True):
        super(ImpactonCalib, self).__init__(product_dir=product_dir,
                                          calib_dir=calib_dir,
                                          ext=ext,
                                          fits_extensions=fits_extensions,
                                          compression=compression)

    def get_frame_name(self, type, filegroup):
        """Return a convenient name for a master frame."""
        if type == 'bias':
            name = 'bias_'
            l = self._bias_name_keywords
        elif type == 'flat':
            name = 'flat_'
            l = self._flat_name_keywords
        elif type == 'science':
            name = ''
            l = self._science_name_keywords
        else:
            raise ValueError('Type {} not supported.'.format(type))
        for i in l:
            v = filegroup.values(i, unique=True)[0]
            name += '{}_'.format(v)
        name = name.replace(' ', '-').strip('_')
        name += self._save_fits_fmt
        return name

    def _compute_night(self, dateobs, timezone):
        t = Time(dateobs, format='isot', scale='utc')
        if timezone is not None:
            timezone = TimezoneInfo(utc_offset=timezone*units.hour)
        # Assume all images where taken in the same night
        dat = t.to_datetime(timezone)
        yyyy = dat.year
        mm = dat.month
        dd = dat.day
        # Assume one night goes from dd 12pm to nextdd-1 12pm
        if dat.hour < 12:
            dd -= 1
        return '{:04d}-{:02d}-{:02d}'.format(yyyy, mm, dd)

    def get_site(self, fg=None):
        """Get the site location, based or not in a filegroup."""
        if fg is None:
            raise ValueError('For this pipeline, you need to specify a '
                             'filegroup for site determination.')

        lat = Angle(fg.values('sitelat', unique=True)[0], unit='degree')
        lon = Angle(fg.values('sitelong', unique=True)[0], unit='degree')
        alt = 0

        return EarthLocation.from_geodetic(lat, lon, alt)

    def get_timezone(self, fg=None):
        """Get the time zone of the observing site."""
        # For IMPACTON, the timezone is -3
        return -3

    def tune_calib_frame(self, type, file, to_calib_filegroup):
        """Reprocess things in calib frames in order to be usable.

        like trimming, binning, etc.
        """
        if file is not None:
            hdu = check_hdu(file, ext=self._save_fits_ext)
            return hdu

        return

class ImpactonStackedPhotometry(StackedPhotometryPipeline):
    photometry_parameters = dict(detect_snr=5,
                                 detect_fwhm=4,
                                 photometry_type='aperture',
                                 r=np.arange(20),
                                 r_in=25,
                                 r_out=30,
                                 solve_photometry_type='montecarlo',
                                 montecarlo_iters=300,
                                 montecarlo_percentage=0.2)
    astrometry_parameters = dict(ra_key='RA',
                                 dec_key='DEC',
                                 plate_scale=0.45,
                                 identify_limit_angle='2 arcsec')
    combine_parameters = dict(method='sum',
                              weights=None,
                              scale=None,
                              mem_limit=1e8,
                              reject=None)
    save_file_name = '{object}_{filter}_{night}.fits'
    save_file_dir = 'stacked_photometry/'

    def __init__(self, product_dir, image_ext=0):
        super(ImpactonStackedPhotometry, self).__init__(product_dir, image_ext)
