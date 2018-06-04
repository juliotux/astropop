from string import Formatter
from collections import namedtuple
import os
import re
import copy

from ...file_manager import FileManager
from ...logger import logger
from ...py_utils import mkdir_p
from ...image_processing.utils import (combine_bias, combine_dark,
                                             combine_flat)
from ...image_processing.ccd_processing import process_image


PipeProd = namedtuple('PipeProd', ['files', 'bias', 'flat', 'dark'])


class SimpleCalibPipeline():
    template = 'LNA_robo40_photometry'
    calib_process_params = {}
    sci_process_params = {}
    _science_group_keywords = []
    _science_select_rules = {}
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

    def __init__(self, product_dir=None, ext=0,
                 fits_extensions=['.fits'], compression=True):
        self.fm = FileManager(ext=ext, fits_extensions=fits_extensions,
                              compression=compression)

        self.save_fm = FileManager(ext=self._save_fits_ext,
                                   fits_extensions=[self._save_fits_fmt])

        if product_dir is not None:
            if not os.path.exists(product_dir):
                mkdir_p(product_dir)
            elif not os.path.isdir(product_dir):
                raise ValueError('Product dir {} not valid!'.format(product_dir))
        self.product_dir = product_dir or os.path.expanduser('~/astropop')

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
        else:
            raise ValueError('Type {} not supported.'.format(type))
        for i in l:
            name += '{}_'.format(filegroup.values(i, unique=True)[0])
        name = name.replace(' ', '-').strip('_')
        name += self._save_fits_fmt
        return name

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
                                     **self.calib_process_params)
                    elif type == 'dark':
                        _bias = self.tune_calib_frame('bias', selected['bias'],
                                                      raw_calib)
                        combine_dark(raw_calib.hdus(), save_file=_file,
                                     master_bias=_bias,
                                     save_compress=self._save_fits_compressed,
                                     **self.calib_process_params)
                    elif type == 'flat':
                        _bias = self.tune_calib_frame('bias', selected['bias'],
                                                      raw_calib)
                        _dark = self.tune_calib_frame('dark', selected['dark'],
                                                      raw_calib)
                        combine_flat(raw_calib.hdus(), save_file=_file,
                                     master_bias=_bias, dark_frame=_dark,
                                     save_compress=self._save_fits_compressed,
                                     **self.calib_process_params)
                    calib_filegroup.add_file(_file)
                else:
                    logger.warn('No raw {} images found. Skipping {} calib.'
                                .format(type, type))
            selected[type] = _file

        for i in ['bias', 'dark', 'flat']:
            _selection(i)

        return PipeProd(sci_filegroup, selected['bias'], selected['dark'],
                        selected['flat'])

    def _process_sci_im(self, files, bias=None, dark=None,
                           flat=None, save_to=None):
        _bias = self.tune_calib_frame('bias', bias, files)
        _dark = self.tune_calib_frame('dark', dark, files)
        _flat = self.tune_calib_frame('flat', flat, files)

        for hdu, name in zip(files.hdus(), files.files):
            if _flat is None or _bias is None:
                raise ValueError('Bias or Flat missing!')
            logger.debug("bias: {}\nflat: {}\ndark: {}".format(bias, dark,
                                                               flat))
            filename = os.path.basename(name)
            filename = os.path.join(save_to, filename)
            yield process_image(hdu, save_to=filename, master_bias=_bias,
                                master_flat=_flat,
                                dark_frame=_dark,
                                save_compressed=self._save_fits_compressed,
                                **self.sci_process_params)

    def get_dirs(self, subfolder=None, fg=None):
        """Return the path to calib_dir and product_dir."""
        calib_dir = os.path.join(self.product_dir, 'calib')
        red_dir = os.path.join(self.product_dir, 'science')
        if subfolder is not None:
            keys = Formatter().parse(self._save_subfolder)
            k = {}
            for i in keys:
                # Assume all have same value
                k[i] = fg.values(i, unique=True)[0]
            subfold = self._save_subfolder.format(**k)
            calib_dir = os.path.join(calib_dir, subfold)
            red_dir = os.path.join(red_dir, subfold)
        mkdir_p(calib_dir)
        mkdir_p(red_dir)
        logger.debug("calib_dir: {}\nproduct_dir: {}".format(calib_dir,
                                                             red_dir))
        return calib_dir, red_dir

    def run(self, raw_dir):
        """Process the data."""
        calib_dir, red_dir = self.get_dirs()

        products = self.pre_run(raw_dir, calib_dir)

        for p in products:
            sci_processed_dir = os.path.join(red_dir, 'calibed_images')
            mkdir_p(sci_processed_dir)
            processed = self._process_sci_im(p.files, p.bias, p.dark, p.flat,
                                             save_to=sci_processed_dir)
            n_tot = len(p.files)
            i = 0
            for hdu in processed:
                i += 1
                logger.info('Processed file {} from {}'.format(i, n_tot))

    def pre_run(self, raw_dir, calib_dir):
        """Pre processing of data. Run before `run` function."""
        if not os.path.isdir(raw_dir):
            raise ValueError('Raw dir {} not valid!'.format(raw_dir))
        raw_dir = os.path.abspath(raw_dir)

        calib_files = self.save_fm.create_filegroup(path=calib_dir)
        raws = self.fm.create_filegroup(path=raw_dir)

        # First step: select science files and groups
        sci_files = self.fm.filtered(raws, **self._science_select_rules)

        prod_list = []
        for g in self.fm.group_by(sci_files, self._science_group_keywords):
            obj = g.values('object', unique=True)[0]
            if re.match('[\w\W]+ acq$', obj) or re.match('[\w\W]+ acq$', obj) :
                logger.info('Acquisition images ignored: {}'.format(g.files))
            else:
                prod_list.append(g)

        products = []
        for sci_fg in prod_list:
            products.append(self.select_calib(sci_fg, calib_files, raws,
                                              calib_dir, raw_dir))
        return products

    def post_run(self, raw_dir):
        """Post processing of data. Run after `run` function."""
