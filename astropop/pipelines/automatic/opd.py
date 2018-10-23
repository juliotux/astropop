from astropy.time import Time, TimezoneInfo
from astropy.coordinates import Angle, EarthLocation
from astropy.nddata.utils import block_replicate
from astropy import units
from astropy.io import fits
import numpy as np

from .common_processing import SimpleCalibPipeline
from .common_photometry import StackedPhotometryPipeline
from ...catalogs import default_catalogs
from ...logger import logger
from ...image_processing.ccd_processing import trim_image


cat_list = {'B': default_catalogs['APASS'],
            'V': default_catalogs['APASS'],
            'R': default_catalogs['GSC2.3'],
            'I': default_catalogs['DENIS']}


class ROBO40Calib(SimpleCalibPipeline):
    template = 'LNA_robo40_photometry'
    _science_group_keywords = ['object', 'program', 'detsec', 'ccdsum',
                               'filter', 'gain']
    _science_select_rules = dict(imagetyp=['OBJECT', 'object'], shutter='OPEN',
                                 site='OPD',
                                 instrume='ASCOM: Apogee AltaU-16M: KAF16803',
                                 telescop='Software Bisque The Sky telescope')
    _science_name_keywords = ['object', 'night', 'filter']
    _bias_select_keywords = ['instrume', 'telescop', 'site', 'gain', 'night',
                             'ccdsum']
    _bias_select_rules = dict(imagetyp=['ZERO', 'zero'], shutter='CLOSE')
    _bias_name_keywords = ['night', 'ccdsum', 'gain']
    _flat_select_keywords = ['instrume', 'telescop', 'site', 'gain',
                             'filter', 'ccdsum', 'night', 'ccdsum']
    _flat_select_rules = dict(imagetyp=['FLAT','flat'], shutter='OPEN')
    _flat_name_keywords = ['night', 'ccdsum', 'gain', 'filter']
    _dark_select_keywords = ['instrume', 'telescop', 'site', 'gain',
                             'ccdsum', 'night']
    _dark_select_rules = dict(imagetyp=['DARK', 'dark'], shutter='CLOSE')
    _dark_name_keywords = ['night', 'ccdsum', 'gain']
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
    astrometry_params = dict(ra_key='RA',
                             dec_key='DEC',
                             pltscl=0.45)
    _save_subfolder = '{program}/{night}'
    plate_scale = 0.45
    _align_method = 'chi2'

    def __init__(self, product_dir=None, calib_dir=None,
                 ext=1, fits_extensions=['.fz'], compression=False):
        super(ROBO40Calib, self).__init__(product_dir=product_dir,
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
        elif type == 'dark':
            name = 'dark_'
            l = self._dark_name_keywords
        elif type == 'science':
            name = ''
            l = self._science_name_keywords
        else:
            raise ValueError('Type {} not supported.'.format(type))
        for i in l:
            v = filegroup.values(i, unique=True)[0]
            if i == 'ccdsum':
                v = v.strip()
                v = v.replace(' ', 'x')
            name += '{}_'.format(v)
        name = name.replace(' ', '-').strip('_')
        name += self._save_fits_fmt
        return name

    def get_platescale(self, file):
        binning = int(file.header['ccdsum'].split()[0])
        return self.plate_scale*binning

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

        lat = Angle(fg.values('latitude', unique=True)[0], unit='degree')
        lon = Angle(fg.values('longitud', unique=True)[0], unit='degree')
        alt = float(fg.values('altitude', unique=True)[0])

        return EarthLocation.from_geodetic(lat, lon, alt)

    def get_timezone(self, fg=None):
        """Get the time zone of the observing site."""
        # For OPD, the timezone is -3
        if fg is not None:
            if fg.values('site', unique=True)[0] != 'OPD':
                raise ValueError('Site not supported by this pipeline.')
        return -3

    def tune_calib_frame(self, type, file, to_calib_filegroup):
        """Reprocess things in calib frames in order to be usable.

        like trimming, binning, etc.
        """
        if file is not None:
            # # Now, with the new observing program, only trimming is needed
            # tocalib = to_calib_filegroup.values('ccdsum',
            #                                     unique=True)[0].split()
            # frame = fits.getval(file, 'ccdsum', ext=self._save_fits_ext).split()
            # rebin = np.array([int(frame[0])/int(tocalib[0]),
            #                   int(frame[1])/int(tocalib[1])], dtype=int)
            #
            # if type == 'bias':
            #     conserve_sum = False
            # else:
            #     conserve_sum = True

            hdu = fits.open(file)[self._save_fits_ext]
            # hdu.data = block_replicate(hdu.data, rebin,
            #                            conserve_sum=conserve_sum)

            if 'detsec' in to_calib_filegroup.summary.colnames:
                slic = to_calib_filegroup.values('detsec', unique=True)[0]
                if slic is not None:
                    logger.debug('slicing {} by {}'.format(type, slic))
                    hdu = trim_image(hdu, slic, fits_convention=True)
            return hdu

        return

    # def post_run(self, raw_dir):
    #     calib_dir, red_dir = self.get_dirs(raw_dir)
    #     fm = FileManager(ext=1, fits_extensions=['.fz'])
    #     sci = fm.create_filegroup(os.path.join(red_dir, 'calibed_images'))
    #     groups = fm.group_by(sci, 'project')


class ROBO40Photometry(StackedPhotometryPipeline):
    photometry_parameters = dict(detect_snr=5,
                                 detect_fwhm=4,
                                 photometry_type='aperture',
                                 r="auto",
                                 r_ann="auto",
                                 solve_photometry_type='montecarlo',
                                 montecarlo_iters=300,
                                 montecarlo_percentage=0.2)
    astrometry_parameters = dict(ra_key='RA',
                                 dec_key='DEC',
                                 identify_limit_angle='2 arcsec')
    combine_parameters = dict(method='sum',
                              weights=None,
                              scale=None,
                              mem_limit=1e8,
                              reject=None)
    save_file_name = '{object}_{filter}_{night}.fits'
    save_file_dir = 'stacked_photometry/'
    plate_scale = 0.45

    def __init__(self, product_dir, image_ext=0):
        super(ROBO40Photometry, self).__init__(product_dir, image_ext)

    def get_platescale(self, file):
        binning = int(file.header['ccdsum'].split()[0])
        return self.plate_scale*binning
