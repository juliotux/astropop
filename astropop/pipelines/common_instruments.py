import copy

from ..pipeline import Instrument
from ...logger import logger


class SimpleCCDCamera(Instrument):
    """Basic tasks for a dummy optical camera.

    This design assumes single-CCD optical cameras, with all data in a single
    directory corresponding to only one night.
    """
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
    _save_fits_fmt = '.fits.gz'

    def load_file(self, fname):
        raise NotImplementedError()

    def save_file(self, fname):
        raise NotImplementedError()

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

    def overscan(self, product=None):
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

    def _frame_name(self, prefix, filegroup, name_rules):
        name = prefix
        for i in name_rules:
            name += '{}_'.format(filegroup.values(i, unique=True)[0])
        name = name.replace(' ', '-').strip('_')
        name += self._save_fits_fmt
        return name

    def _create_calib_frame(self, type, raw_filegroup, name_rules, select_rules,
                            create_calib_callback=None, **kwargs):
        """Splited function to create calib frames"""
        # create_calib_callback is a function, to be set, that recieve the
        # arguments (type, raw_files, save_name, **kwargs), process images and
        # create calibration frames, returning the full path to the saved file
        if create_calib_callback is None:
            logger.error('No create_calib_callback set. See documentation for '
                         'this function.')
            return None
        _files = raw_filegroup.filtered(**select_rules)
        if len(_files) == 0:
            logger.error('No raw files were found to create {} calib frame.'
                         .format(type))
            return None
        name = self._frame_name('{}_'.format(type), _files, name_rules)
        fname = create_calib_callback(type, _files.files, name, **kwargs)
        return fname

    def select_calibs(self, product, raw_filegroup, calib_filegroup,
                      create_calib_callback=None):
        """Select the calibration frames for a product."""
        product.bias = None
        product.flat = None
        product.dark = None

        def _selection(type):
            if type == 'bias':
                rules = self._bias_select_rules
                keywords = self._bias_select_keywords
                name_rules = self._bias_name_keywords
            elif type == 'dark':
                rules = self._dark_select_rules
                keywords = self._dark_select_keywords
                name_rules = self._dark_name_keywords
            elif type == 'flat':
                rules = self._flat_select_rules
                keywords = self._flat_select_keywords
                name_rules = self._flat_name_keywords

            _select = copy.copy(rules)
            _dict = {}
            for i in keywords:
                val = product.filegroup.values(i, unique=True)
                if len(val) > 1:
                    raise ValueError('Could not select {}. '
                                     '{} keyword do not have unique value.'
                                     .format(type, i))
                _dict[i] = val[0]
            _select.update(_dict)
            logger.debug("Selecting {} with rules {}".format(type, _select))
            _calib = calib_filegroup.filtered(**_select)
            if len(_calib) == 1:
                product[type] = _calib.files[0]
                logger.info('Using {} from disk: {}'.format(type,
                                                            product[type]))
            elif len(_calib) == 0:
                logger.info('No reduced {} found. Creating a master frame.'
                            .format(type))
                prev_calibs = {'bias': product.bias, 'flat': product.flat,
                               'dark': product.dark}
                name = self._create_calib_frame(type, raw_filegroup, name_rules,
                                                _select, create_calib_callback,
                                                **prev_calibs, instrument=self)
                product[type] = name
                calib_filegroup.add_file(name)
            else:
                logger.error('Multiple files matches this rule. Skipping.')

        for type in ['bias', 'dark', 'flat']:
            _selection(type)

    def _use_product(self, filegroup):
        """Check if a product has to be excluded. Like 'acq' frames."""
        return True

    def create_products(self, product_manager, raw_filegroup):
        """Select science files based on sicence_select_rules and
        science_group_keywords."""
        logger.info('Selecting science files with rules: {}'
                     .format(self._science_select_rules))
        sci_files = raw_filegroup.filtered(**self._science_select_rules)

        logger.info('Grouping science files with: {}'
                     .format(self._science_group_keywords))
        for g in sci_files.group_by(sci_files, self._science_group_keywords):
            if self._use_product(g):
                product_manager.create_product(raw_filegroup=g,
                                               raw_files=g.files,
                                               ccddata=None)
        logger.info('{} groups created'
                    .format(product_manager.number_of_products))
