from astropy.time import Time, TimezoneInfo
from astropy.coordinates import Angle, EarthLocation
from astropy import units
from astropy.io import fits
import numpy as np

from .common_processing import SimpleCalibPipeline
from .common_photometry import StackedPhotometryPipeline
from .common_polarimetry import PolarimetryPileline
from ...logger import logger
from ...image_processing.ccd_processing import trim_image
from ..polarimetry_scripts import dualbeam_polarimetry

class Polima2Calib(SimpleCalibPipeline):
    template = 'Polima2_Calib'
    _science_group_keywords = ['object', 'instrume', 'origin', 'observat',
                               'telescop', 'proj_id', 'detector', 'ccdsum',
                               'date-obs', 'filter', 'pol2lin', 'ccdmode',
                               'ampname']
    _science_select_rules = dict(imagetyp=['image', 'IMAGE',
                                           'object', 'OBJECT'])
    _science_name_keywords = ['object', 'date-obs', 'filter', 'pol2lin']
    _bias_select_keywords = ['ccdmode', 'ampname', 'date-obs', 'ccdsum',
                             'detector', 'telescop', 'proj_id', 'instrume']
    _bias_select_rules = dict(imagetyp=['ZERO', 'zero'])
    _bias_name_keywords = ['detector', 'night', 'ccdsum', 'ampname']
    _flat_select_keywords = ['ccdmode', 'ampname', 'date-obs', 'ccdsum',
                             'detector', 'telescop', 'proj_id', 'instrume',
                             'filter']
    _flat_select_rules = dict(imagetyp=['FLAT','flat'])
    _flat_name_keywords = ['detector', 'night', 'ccdsum', 'ampname',
                           'filter']
    _dark_select_keywords = ['ccdmode', 'ampname', 'ccdsum',
                             'detector', 'telescop', 'proj_id', 'instrume']
    _dark_select_rules = dict(imagetyp=['DARK', 'dark'])
    _dark_name_keywords = ['detector', 'night', 'ccdsum', 'ampname']
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
                             pltscl=0.22)
    _save_subfolder = 'Polima2/{date-obs}'
    plate_scale = 0.22
    _align_method = 'chi2'
    # _save_fits_compressed = False
    # _save_fits_ext = 0
    _save_fits_fmt = '.fits.gz'

    def __init__(self, product_dir=None, calib_dir=None,
                 ext=0, fits_extensions=['.fit'], compression=False):
        super(Polima2Calib, self).__init__(product_dir=product_dir,
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

    def get_site(self, fg):
        """Get the site location, based or not in a filegroup."""
        lat = fg.values('latitude')[0]
        lon = fg.values('longitud')[0]
        alt = fg.values('altitude')[0]
        return EarthLocation.from_geodetic(lon, lat, alt)

    def get_timezone(self, fg=None):
        """Get the time zone of the observing site."""
        return fg.values('timezone')[0]

    def tune_calib_frame(self, type, file, to_calib_filegroup):
        """Reprocess things in calib frames in order to be usable.

        like trimming, binning, etc.
        """
        if file is not None:
            hdu = fits.open(file)[self._save_fits_ext]
            return hdu
        return

# class IAGPOLPipeline(PolarimetryPileline):
#     photometry_parameters = dict(detect_snr=5,
#                                  detect_fwhm=4,
#                                  photometry_type='aperture',
#                                  r='auto',
#                                  r_ann='auto',
#                                  solve_photometry_type='montecarlo',
#                                  montecarlo_iters=300,
#                                  montecarlo_percentage=0.2)
#     astrometry_parameters = dict(ra_key='RA',
#                                  dec_key='DEC',
#                                  identify_limit_angle='2 arcsec')
#     save_file_name = '{object}_{analyzer}_{filter}_{night}.fits'
#     save_file_dir = 'polarimetry/'
#     filter_key = 'fil-pos'
#     lamina_key = 'lam-pos'
#     retarder_key = 'retarder'
#     analyzer_key = 'analyzer'
#     step = 22.5
#     standard_catalogs = {'U': 'Simbad',
#                          'B': 'UCAC4',
#                          'V': 'UCAC4',
#                          'R': 'UCAC5',
#                          'I': 'DENIS'}
#     plate_scale = np.array([0.31, 0.65])
#     _align_method = 'chi2'
#
#     def __init__(self, product_dir, image_ext=0):
#         super(IAGPOLPipeline, self).__init__(product_dir, image_ext)
#
#     def _get_lamina(self, filegroup):
#         ret = filegroup.values(self.lamina_key)
#         if None in ret:
#             return np.arange(len(filegroup.summary))
#         else:
#             try:
#                 return np.array([int(i) for i in ret])
#             except ValueError:
#                 # Some positions may be in hexa
#                 return np.array([int(i, 16) for i in ret])
#
#     def get_psi(self, filegroup):
#         """Get analyzer os retarder positions in degrees."""
#         return self.step*self._get_lamina(filegroup)
#
#     def get_retarder(self, filegroup):
#         """The retarder identifier, used for what polarimetry will perform."""
#         return filegroup.values(self.retarder_key, unique=True)[0]
#
#     def get_analyzer(self, filegroup):
#         """The analyzer identifier, for polarimetry function selection."""
#         return filegroup.values(self.analyzer_key, unique=True)[0]
#
#     def get_filter(self, filegroup):
#         filt = filegroup.values(self.filter_key, unique=True)[0]
#         try:
#             filt = 'UBVRI'[int(filt)]
#         except ValueError:
#             pass
#         return filt
#
#     def validate_data(self, filegroup):
#         """Validate if it's everything ok with the data."""
#         if len(filegroup.summary) < 4:
#             raise ValueError('Not enough calibed images to process.')
#         return True
#
#     def calculate_polarimetry(self, fg, analyzer, retarder, psi, catalog,
#                               sci_catalog, astrometry, **kwargs):
#         # instrument specific tasks
#
#         if analyzer in ['A0', 'A2']:
#             results = dualbeam_polarimetry(fg.hdus(), psi,
#                                            identify_catalog=catalog,
#                                            positions=self._get_lamina(fg),
#                                            retarder=retarder,
#                                            analyzer=analyzer,
#                                            match_pairs_tolerance=1.0,
#                                            calculate_mode='fit',
#                                            plate_scale=self.plate_scale,
#                                            science_catalog=sci_catalog,
#                                            **self.astrometry_parameters,
#                                            **self.photometry_parameters,
#                                            **kwargs)
#
#         return results
