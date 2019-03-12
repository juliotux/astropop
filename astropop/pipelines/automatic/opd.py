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
from ..photometry_scripts import process_calib_photometry


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

        return EarthLocation.from_geodetic(lon, lat, alt)

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
    save_file_name = '{program}_{object}_{filter}_{night}.fits'
    save_file_dir = 'stacked_photometry/'
    plate_scale = 0.45
    standard_catalogs = {'U': 'Simbad',
                         'B': 'UCAC4',
                         'V': 'UCAC4',
                         'R': 'UCAC5',
                         'I': 'DENIS'}

    def __init__(self, product_dir, image_ext=0):
        super(ROBO40Photometry, self).__init__(product_dir, image_ext)

    def get_platescale(self, file):
        binning = int(file.header['ccdsum'].split()[0])
        return self.plate_scale*binning


class BCCalib(SimpleCalibPipeline):
    template = 'LNA_BC_calib'
    _science_group_keywords = ['object', 'acqmode', 'imgrect',
                               'subrect', 'readtime', 'emrealgn', 'preamp',
                               'serno', 'camgain', 'filter', 'gain', 'hbin',
                               'vbin', 'outptamp', 'telescop', 'analyzer']
    _science_select_rules = dict(imagetyp=['OBJECT', 'object'])
    _science_name_keywords = ['object', 'night', 'filter', 'outptamp',
                              'analyzer']
    _bias_select_keywords = ['acqmode', 'imgrect', 'night',
                             'subrect', 'readtime', 'emrealgn', 'preamp',
                             'serno', 'camgain', 'gain', 'hbin',
                             'vbin', 'outptamp', 'telescop']
    _bias_select_rules = dict(imagetyp=['ZERO', 'zero', 'BIAS', 'bias'])
    _bias_name_keywords = ['night', 'hbin', 'vbin', 'gain', 'outptamp']
    _flat_select_keywords = ['acqmode', 'imgrect', 'filter', 'analyzer',
                             'subrect', 'readtime', 'emrealgn', 'preamp',
                             'serno', 'camgain', 'gain', 'hbin',
                             'vbin', 'outptamp', 'telescop', 'night']
    _flat_select_rules = dict(imagetyp=['FLAT','flat'])
    _flat_name_keywords = ['night', 'hbin', 'vbin', 'gain', 'filter',
                           'analyzer', 'outptamp']
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
    plate_scale = 0.32
    _align_method = 'chi2'
    _save_fits_compressed = False
    _save_fits_ext = 0
    _save_fits_fmt = '.fits.gz'

    def __init__(self, product_dir=None, calib_dir=None,
                 ext=0, fits_extensions=['.fits', '.fits.gz'],
                 compression=False):
        super(BCCalib, self).__init__(product_dir=product_dir,
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
        binning = int(file.header['vbin'].split()[0])
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
        return EarthLocation.from_geodetic('-45d34m57s', '-22d32m04s', 1864)

    def get_timezone(self, fg=None):
        """Get the time zone of the observing site."""
        return -3

    def tune_calib_frame(self, type, file, to_calib_filegroup):
        """Reprocess things in calib frames in order to be usable.

        like trimming, binning, etc.
        """
        if file is not None:
            hdu = fits.open(file)[self._save_fits_ext]
            return hdu
        return

class IAGPOLPipeline(PolarimetryPileline):
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
    filter_key = 'fil-pos'
    lamina_key = 'lam-pos'
    retarder_key = 'retarder'
    analyzer_key = 'analyzer'
    step = 22.5
    standard_catalogs = {'U': 'Simbad',
                         'B': 'UCAC4',
                         'V': 'UCAC4',
                         'R': 'UCAC5',
                         'I': 'DENIS'}
    plate_scale = np.array([0.31, 0.65])
    _align_method = 'chi2'

    def __init__(self, product_dir, image_ext=0):
        super(IAGPOLPipeline, self).__init__(product_dir, image_ext)

    def _get_lamina(self, filegroup):
        ret = filegroup.values(self.lamina_key)
        if None in ret:
            return np.arange(len(filegroup.summary))
        else:
            try:
                return np.array([int(i) for i in ret])
            except ValueError:
                # Some positions may be in hexa
                return np.array([int(i, 16) for i in ret])

    def get_psi(self, filegroup):
        """Get analyzer os retarder positions in degrees."""
        return self.step*self._get_lamina(filegroup)

    def get_retarder(self, filegroup):
        """The retarder identifier, used for what polarimetry will perform."""
        return filegroup.values(self.retarder_key, unique=True)[0]

    def get_analyzer(self, filegroup):
        """The analyzer identifier, for polarimetry function selection."""
        return filegroup.values(self.analyzer_key, unique=True)[0]

    def get_filter(self, filegroup):
        filt = filegroup.values(self.filter_key, unique=True)[0]
        if filt is None:
            filt = filegroup.values('filter', unique=True)[0]
        try:
            filt = 'UBVRI'[int(filt)]
        except Exception as e:
            pass
        logger.debug('Filter: {}'.format(filt))
        return filt

    def validate_data(self, filegroup):
        """Validate if it's everything ok with the data."""
        if len(filegroup.summary) < 4:
            raise ValueError('Not enough calibed images to process. {} images'
                             ' found'.format(len(filegroup.summary)))
        return True

    def calculate_polarimetry(self, fg, analyzer, retarder, psi, catalog,
                              sci_catalog, astrometry, **kwargs):
        # instrument specific tasks

        if analyzer in ['A0', 'A2']:
            results = dualbeam_polarimetry(fg.hdus(), psi,
                                           identify_catalog=catalog,
                                           positions=self._get_lamina(fg),
                                           retarder=retarder,
                                           analyzer=analyzer,
                                           match_pairs_tolerance=1.0,
                                           calculate_mode='fit',
                                           plate_scale=self.plate_scale,
                                           science_catalog=sci_catalog,
                                           **self.astrometry_parameters,
                                           **self.photometry_parameters,
                                           **kwargs)
        elif analyzer in [None, 'A3']:
            stacked = None
            for i in fg.hdus():
                if stacked is None:
                    stacked = i
                else:
                    s = hdu_shift_images([stacked, i],
                                         method=self._align_method)[1]
                    stacked = imarith(stacked, s, '+')

            filt = self.get_filter(fg)
            cat = self.select_catalog(filt)

            wcs = WCS(stacked.header, relax=True)
            if '' in wcs.wcs.ctype or astrometry:
                wcs = None

            plate_scale = self.get_platescale(stacked)
            phot, wcs = process_calib_photometry(stacked,
                                                 science_catalog=science_catalog,
                                                 identify_catalog=identify_catalog,
                                                 filter=filt, wcs=wcs,
                                                 return_wcs=True,
                                                 plate_scale=plate_scale,
                                                 **self.photometry_parameters,
                                                 **self.astrometry_parameters)

            apertures = np.unique(phot['aperture'])
            if len(apertures) > 1:
                selected_aperture = 0
                snr = 0
                for g in phot.group_by('aperture').groups:
                    g_snr = np.sum(g['flux']/g['flux_error'])
                    if g_snr > snr:
                        selected_aperture = g['aperture'][0]
                        snr = g_snr

                phot = phot[phot['aperture'] == selected_aperture]
                phot = phot.as_array()

            phot.meta['astropop n_images'] = len(prod.calibed_files)
            phot.meta['astropop night'] = prod.files.values('night',
                                                            unique=True)[0]
            phot.meta['astropop filter'] = filt
            stacked.header['astropop n_images'] = len(prod.calibed_files)
            stacked.header['astropop night'] = prod.files.values('night',
                                                                 unique=True)[0]
            stacked.header['astropop filter'] = filt

            header=stacked.header
            if wcs is not None:
                header.update(wcs.to_header(relax=True))

            result = (stacked, phot, wcs)

        return results
