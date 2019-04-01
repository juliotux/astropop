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
    template = 'Polima2_calib'
    _science_group_keywords = ['object', 'acqmode', 'imgrect',
                               'subrect', 'readtime', 'emrealgn', 'preamp',
                               'serno', 'camgain', 'filter', 'gain', 'hbin',
                               'vbin', 'outptamp', 'telescop', 'analyzer']
    _science_select_rules = dict(imagetyp=['OBJECT', 'object'])
    _science_name_keywords = ['object', 'night', 'filter', 'outptamp']
    _bias_select_keywords = ['acqmode', 'imgrect', 'night',
                             'subrect', 'readtime', 'emrealgn', 'preamp',
                             'serno', 'camgain', 'gain', 'hbin',
                             'vbin', 'outptamp', 'telescop']
    _bias_select_rules = dict(imagetyp=['ZERO', 'zero'])
    _bias_name_keywords = ['night', 'hbin', 'vbin', 'gain', 'outptamp']
    _flat_select_keywords = ['acqmode', 'imgrect', 'filter', 'analyzer',
                             'subrect', 'readtime', 'emrealgn', 'preamp',
                             'serno', 'camgain', 'gain', 'hbin',
                             'vbin', 'outptamp', 'telescop', 'night']
    _flat_select_rules = dict(imagetyp=['FLAT','flat'])
    _flat_name_keywords = ['night', 'hbin', 'vbin', 'gain', 'filter', 'outptamp']
    _dark_select_keywords = ['acqmode', 'imgrect', 'night',
                             'subrect', 'readtime', 'emrealgn', 'preamp',
                             'serno', 'camgain', 'gain', 'hbin',
                             'vbin', 'outptamp', 'telescop']
    _dark_select_rules = dict(imagetyp=['DARK', 'dark'])
    _dark_name_keywords = ['night', 'hbin', 'vbin', 'gain', 'outptamp']
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
    _save_subfolder = 'BC/{night}'
    plate_scale = 0.45
    _align_method = 'chi2'
    _save_fits_compressed = False
    _save_fits_ext = 0
    _save_fits_fmt = '.fits.gz'

    def __init__(self, product_dir=None, calib_dir=None,
                 ext=0, fits_extensions=['.fits', '.fits.gz'],
                 compression=False):
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
            name += '{}_'.format(v)
        name = name.replace(' ', '-').strip('_')
        name += self._save_fits_fmt
        return name

    def get_platescale(self, file):
        binning = int(file.header['ccdsum'].split()[0])
        return self.plate_scale*binning

    def tune_calib_frame(self, type, file, to_calib_filegroup):
        """Reprocess things in calib frames in order to be usable.

        like trimming, binning, etc.
        """
        if file is not None:
            hdu = fits.open(file)[self._save_fits_ext]
            return hdu
        return

class Polima2Pipeline(PolarimetryPileline):
    photometry_parameters = dict(detect_snr=5,
                                 detect_fwhm=4,
                                 photometry_type='aperture',
                                 r='auto',
                                 r_ann='auto',
                                 solve_photometry_type='montecarlo',
                                 montecarlo_iters=300,
                                 montecarlo_percentage=0.2)
    astrometry_parameters = dict(ra_key='RA',
                                 dec_key='DEC',
                                 identify_limit_angle='2 arcsec')
    save_file_name = '{object}_{analyzer}_{filter}_{night}.fits'
    save_file_dir = 'polarimetry/'
    filter_key = 'filter'
    lamina_key = 'polangle'
    retarder_key = 'retarder'
    analyzer_key = 'pol2lin'
    standard_catalogs = {'U': 'Simbad',
                         'B': 'UCAC4',
                         'V': 'UCAC4',
                         'R': 'UCAC5',
                         'I': 'DENIS'}
    plate_scale = np.array([0.31, 0.65])
    _align_method = 'chi2'

    def __init__(self, product_dir, image_ext=0):
        super(Polima2Pipeline, self).__init__(product_dir, image_ext)

    def get_psi(self, filegroup):
        """Get analyzer os retarder positions in degrees."""
        return filegroup.values(self.lamina_key)

    def get_retarder(self, filegroup):
        """The retarder identifier, used for what polarimetry will perform."""
        return 'half'

    def get_analyzer(self, filegroup):
        """The analyzer identifier, for polarimetry function selection."""
        return filegroup.values(self.analyzer_key, unique=True)[0]

    def get_filter(self, filegroup):
        return filegroup.values(self.filter_key, unique=True)[0]

    def validate_data(self, filegroup):
        """Validate if it's everything ok with the data."""
        if len(filegroup.summary) < 4:
            raise ValueError('Not enough calibed images to process.')
        return True

    def calculate_polarimetry(self, fg, analyzer, retarder, psi, catalog,
                              sci_catalog, astrometry, **kwargs):
        # instrument specific tasks
        results = dualbeam_polarimetry(fg.hdus(), psi,
                                       identify_catalog=catalog,
                                       retarder=retarder,
                                       analyzer=analyzer,
                                       match_pairs_tolerance=1.0,
                                       calculate_mode='fit',
                                       plate_scale=self.plate_scale,
                                       science_catalog=sci_catalog,

                                       **self.astrometry_parameters,
                                       **self.photometry_parameters,
                                       **kwargs)

        return results
