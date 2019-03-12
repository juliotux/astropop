from string import Formatter
import os
import re
import copy
import numpy as np

from ...file_manager import FileManager
from ...logger import logger
from ...py_utils import mkdir_p
from ...image_processing.utils import (combine_bias, combine_dark,
                                             combine_flat)
from ...image_processing.ccd_processing import process_image
from ...image_processing.imarith import imarith
from ...image_processing.register import hdu_shift_images
from ...astrometry.astrometrynet import solve_astrometry_hdu
from ...fits_utils import save_hdu


class PipeProd():
    def __init__(self, files, bias, flat, dark=None, calibed_files=None,
                 sci_result=None):
        self.files = files
        self.bias = bias
        self.flat = flat
        self.dark = dark
        self.calibed_files = calibed_files
        self.sci_result = sci_result


class SimpleCalibPipeline():
    template = 'calib_base'
    calib_process_params = {}
    sci_process_params = {}
    astrometry_params = {}
    _science_group_keywords = []
    _science_select_rules = {}
    _science_name_keywords = []
    _bias_select_keywords = []
    _bias_select_rules = {}
    _bias_name_keywords = []
    _flat_select_keywords = []
    _flat_select_rules = {}
    _flat_name_keywords = []
    _dark_select_keywords = []
    _dark_select_rules = {}
    _dark_name_keywords = []
    _product_store = []
    _save_subfolder = None
    _save_fits_ext = 1
    _save_fits_fmt = '.fz'
    _save_fits_compressed = True
    _align_method = 'chi2'
    plate_scale = None

    def __init__(self, product_dir=None, calib_dir=None, ext=0,
                 fits_extensions=['.fits'], compression=True):
        self.fm = FileManager(ext=ext, fits_extensions=fits_extensions,
                              compression=compression)

        self.save_fm = FileManager(ext=self._save_fits_ext,
                                   fits_extensions=[self._save_fits_fmt])

        if product_dir is not None:
            if not os.path.exists(product_dir):
                mkdir_p(product_dir)
            elif not os.path.isdir(product_dir):
                raise ValueError('Product dir {} not valid!'
                                 .format(product_dir))
        self.product_dir = product_dir or os.path.expanduser('~/astropop')

        if calib_dir is not None:
            if not os.path.exists(calib_dir):
                mkdir_p(calib_dir)
            elif not os.path.isdir(calib_dir):
                raise ValueError('Calibration dir {} not valid!'
                                 .format(calib_dir))
        self.calib_dir = calib_dir or os.path.join(self.product_dir, 'calib')

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
            name += '{}_'.format(filegroup.values(i, unique=True)[0])
        name = name.replace(' ', '-').strip('_')
        name += self._save_fits_fmt
        return name

    def get_platescale(self, file):
        if self.plate_scale is None and \
           'pltsccl' in self.astrometry_parameters:
            self.plate_scale = self.astrometry_parameters.pop('pltscl', None)
        return self.plate_scale

    def tune_calib_frame(self, type, file, to_calib_filegroup):
        """Reprocess things in calib frames in order to be usable.

        like trimming, binning, etc.
        """
        raise NotImplementedError()

    def get_site(self, fg=None):
        """Get the site location, based or not in a filegroup."""
        raise NotImplementedError()

    def get_timezone(self, fg=None):
        """Get the time zone of the observing site."""
        raise NotImplementedError()

    def _compute_night(self, dateobs, timezone):
        raise NotImplementedError()

    def get_night(self, fg, timezone=None, iter=False):
        """Get date-obs(UT), lat, lon and alt and return the observing night.

        If iter is False, just the first value will be computed. Else an array
        with all values will be returned.
        """
        if iter:
            return [self._compute_night(dateobs, timezone) for
                    dateobs in fg.values('date-obs')]
        else:
            return self._compute_night(fg.values('date-obs')[0], timezone)


    def select_calib(self, sci_filegroup, calib_filegroup, raw_filegroup,
                     calib_dir, raw_dir):
        """Select, group and process calib frames."""
        selected = {'bias': None, 'flat': None, 'dark': None}

        # check if 'night' keyword already in headers
        add_key = {}
        for h in raw_filegroup.headers():
            if 'night' not in h.keys():
                # They should have the same values
                add_key = {'night': sci_filegroup.values('night',
                                                         unique=True)[0]}
                break

        def _selection(type):
            if type == 'bias':
                rules = self._bias_select_rules
                keywords = self._bias_select_keywords
            elif type == 'dark':
                rules = self._dark_select_rules
                keywords = self._dark_select_keywords
            elif type == 'flat':
                rules = self._flat_select_rules
                keywords = self._flat_select_keywords

            _select = copy.copy(rules)
            _dict = {}
            for i in keywords:
                val = sci_filegroup.values(i, unique=True)
                if len(val) > 1:
                    raise ValueError('Could not select {}. '
                                     '{} keyword do not have unique value.'
                                     .format(type, i))
                _dict[i] = val[0]
            _select.update(_dict)

            logger.debug("Selecting {} with rules {}".format(type, _select))

            _calib = self.save_fm.filtered(calib_filegroup, **_select)
            _file = None
            if len(_calib) > 0:
                # Assume the first file is the correct. In principle, just one
                # should exists
                _file = _calib.files[0]
                logger.info('Loading {} from disk: {}'.format(type, _file))
            else:
                logger.info('No reduced {} found. Creating a master frame.'
                            .format(type))
                raw_calib = self.fm.filtered(raw_filegroup, **_select)
                if len(raw_calib) > 0:
                    _file = self.get_frame_name(type, raw_calib)
                    _file = os.path.join(calib_dir, _file)
                    if type == 'bias':
                        combine_bias(raw_calib.hdus(), save_file=_file,
                                     save_compress=self._save_fits_compressed,
                                     add_keywords=add_key,
                                     **self.calib_process_params)
                    elif type == 'dark':
                        logger.debug('bias: {}'.format(selected['bias']))
                        _bias = self.tune_calib_frame('bias', selected['bias'],
                                                      raw_calib)
                        combine_dark(raw_calib.hdus(), save_file=_file,
                                     master_bias=_bias,
                                     save_compress=self._save_fits_compressed,
                                     add_keywords=add_key,
                                     **self.calib_process_params)
                    elif type == 'flat':
                        logger.debug('bias: {}'.format(selected['bias']))
                        logger.debug('dark: {}'.format(selected['dark']))
                        _bias = self.tune_calib_frame('bias', selected['bias'],
                                                      raw_calib)
                        _dark = self.tune_calib_frame('dark', selected['dark'],
                                                      raw_calib)
                        combine_flat(raw_calib.hdus(), save_file=_file,
                                     master_bias=_bias, dark_frame=_dark,
                                     save_compress=self._save_fits_compressed,
                                     add_keywords=add_key,
                                     **self.calib_process_params)
                    logger.debug('adding file {} to calibed_filegroup'
                                 .format(_file))
                    calib_filegroup.add_file(_file)
                else:
                    logger.warn('No raw {} images found. Skipping {} calib.'
                                .format(type, type))
            selected[type] = _file

        for type in ['bias', 'dark', 'flat']:
            _selection(type)

        return PipeProd(files=sci_filegroup,
                        bias=selected['bias'],
                        flat=selected['flat'],
                        dark=selected['dark'])

    def _process_sci_im(self, files, bias=None, dark=None,
                        flat=None, save_to=None, astrometry=False):
        _bias = self.tune_calib_frame('bias', bias, files)
        _dark = self.tune_calib_frame('dark', dark, files)
        _flat = self.tune_calib_frame('flat', flat, files)

        if _flat is None or _bias is None:
            logger.error('Bias or Flat missing! Skipping product.')
            return ([], [])

        for hdu, name, night in zip(files.hdus(), files.files,
                                    files.values('night')):
            logger.info("bias: {}".format(bias))
            logger.info("dark: {}".format(dark))
            logger.info("flat: {}".format(flat))
            if save_to is not None:
                filename = os.path.basename(name)
                filename = os.path.join(save_to, filename)
            else:
                filename = None

            if 'night' not in hdu.header.keys():
                hdu.header['night'] = night

            try:
                yield (process_image(hdu, save_to=filename, master_bias=_bias,
                                     master_flat=_flat,
                                     dark_frame=_dark,
                                     save_compressed=self._save_fits_compressed,
                                     overwrite=True,
                                     **self.sci_process_params), filename)
            except Exception as e:
                logger.error('Image not process due to: {}'.format(e))

            if 'file' in hdu.fileinfo():
                hdu.fileinfo()['file'].close()


    def _solve_astrometry(self, hdu):
        plate_scale = self.get_platescale(hdu)
        try:
            params = copy.copy(self.astrometry_params)
            params['pltscl'] = plate_scale
            solved = solve_astrometry_hdu(hdu, return_wcs=True,
                                          image_params=params)
        except:
            solved = None
        if solved is None and 'pltscl' in self.astrometry_params.keys():
            # try with 2 times the plate scale (2x2 binning)
            try:
                params = copy.copy(self.astrometry_params)
                params['pltscl'] = plate_scale
                params['pltscl'] = 2*params['pltscl']
                solved = solve_astrometry_hdu(hdu, return_wcs=True,
                                              image_params=params)
                logger.info('Image astrometry solved.')
                hdu.header['hierarch astrometry.net solved'] = True
            except Exception as e:
                logger.warn('Astrometry not solved! {}'.format(e))
                hdu.header['hierarch astrometry.net solved'] = False

        if solved is not None:
            hdu.header.update(solved.to_header(relax=True))

        return hdu

    def get_dirs(self, subfolder=None, fg=None):
        """Return the path to calib_dir and product_dir."""
        calib_dir = self.calib_dir
        red_dir = os.path.join(self.product_dir, 'science')
        if subfolder is not None:
            keys = Formatter().parse(self._save_subfolder)
            k = {}
            for i in keys:
                # Assume all have same value
                _, i, _, _ = i
                k[i] = fg.values(i, unique=True)[0]
            subfold = self._save_subfolder.format(**k)
            calib_dir = os.path.join(calib_dir, subfold)
            red_dir = os.path.join(red_dir, subfold)
        logger.debug("calib_dir: {}   product_dir: {}".format(calib_dir,
                                                              red_dir))
        return calib_dir, red_dir

    def run(self, raw_dir, stack_images=False, save_calibed=True,
            astrometry=False):
        """Process the data."""
        calib_dir, red_dir = self.get_dirs()
        mkdir_p(calib_dir)

        products = self.pre_run(raw_dir, calib_dir)

        for p in products:
            p_files = []
            if self._save_subfolder:
                # _, red_dir = self.get_dirs(self._save_subfolder, p.files)
                _, sci_processed_dir = self.get_dirs(self._save_subfolder,
                                                     p.files)
            # sci_processed_dir = os.path.join(red_dir, 'calibed_images')
            mkdir_p(sci_processed_dir)
            if not save_calibed:
                save_to = None
            else:
                save_to = os.path.join(sci_processed_dir, 'calibed_images')
                mkdir_p(save_to)
            processed = self._process_sci_im(p.files, p.bias, p.dark, p.flat,
                                             save_to=save_to)
            n_tot = len(p.files)
            i = 0
            if stack_images:
                acumm = None
                footp = None
                logger.info('Stacking images.')
            for hdu, fname in [n for n in processed if n[0] is not None]:
                i += 1
                p_files.append(fname)
                logger.info('Processed file {} from {}'.format(i, n_tot))
                if stack_images:
                    if acumm == None:
                        acumm = hdu
                        footp = np.ones_like(hdu.data)
                    else:
                        hdu = hdu_shift_images([acumm, hdu],
                                               method=self._align_method,
                                               footprint=True)[1]
                        acumm = imarith(acumm, hdu, '+')
                        footp += hdu.footprint

            if stack_images and acumm is not None:
                name = self.get_frame_name('science', p.files)
                name = os.path.join(sci_processed_dir, name)
                if astrometry:
                    acumm = self._solve_astrometry(acumm)
                acumm = imarith(acumm, footp/i, '/')  # normalize borders
                acumm.header['hierarch stacked images'] = i
                save_hdu(acumm, name, compress=self._save_fits_compressed,
                         overwrite=True)
                p_files = [name]

            p.calibed_files = p_files

        return products

    def pre_run(self, raw_dir, calib_dir):
        """Pre processing of data. Run before `run` function."""
        if not os.path.isdir(raw_dir):
            raise ValueError('Raw dir {} not valid!'.format(raw_dir))
        raw_dir = os.path.abspath(raw_dir)
        logger.info('Processing Raw Folder: {}'.format(raw_dir))

        logger.debug('Creating file groups for RAW and Calib files.')
        calib_files = self.save_fm.create_filegroup(path=calib_dir)
        raws = self.fm.create_filegroup(path=raw_dir)

        # First step: select science files and groups
        logger.debug('Creating file groups for Science files.')
        sci_files = self.fm.filtered(raws, **self._science_select_rules)

        logger.debug('Computing night')
        if 'night' not in sci_files.summary.colnames:
            tz = self.get_timezone(sci_files)
            night = self.get_night(sci_files, tz, iter=False)
            raws.add_column('night', night)
            sci_files = self.fm.filtered(raws, **self._science_select_rules)

        logger.debug('Grouping science files.')
        prod_list = []
        for g in self.fm.group_by(sci_files, self._science_group_keywords):
            obj = g.values('object', unique=True)[0]
            if re.match('[\w\W]+ acq$', obj) or re.match('[\w\W]+ acq$', obj) :
                logger.info('Acquisition images ignored: {}'.format(g.files))
            else:
                prod_list.append(g)

        logger.debug('Processing products.')
        products = []
        for sci_fg in prod_list:
            products.append(self.select_calib(sci_fg, calib_files, raws,
                                              calib_dir, raw_dir))
        return products

    def post_run(self, raw_dir):
        """Post processing of data. Run after `run` function."""
