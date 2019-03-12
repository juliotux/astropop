# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from string import Formatter

from ..polarimetry_scripts import dualbeam_polarimetry
from ...py_utils import process_list, check_iterable, mkdir_p
from ...py_utils import check_iterable, mkdir_p
from ...fits_utils import check_hdu
from ...catalogs import default_catalogs, ASCIICatalogClass
from ...image_processing.register import hdu_shift_images
from ...logger import logger
from ...file_manager import FileManager


class PolarimetryPileline():
    photometry_parameters = {}
    astrometry_parameters = {}
    save_file_name = '{object}_{analyzer}_{filter}_{night}.fits'
    save_file_dir = 'stacked_photometry/'
    filter_key = 'filter'
    standard_catalogs = {'U': 'Simbad',
                         'B': 'UCAC4',
                         'V': 'UCAC4',
                         'R': 'UCAC5',
                         'I': 'DENIS'}
    plate_scale = None
    _align_method = 'chi2'

    def __init__(self, product_dir, image_ext=0):
        self.prod_dir = product_dir
        self.image_ext = image_ext

    def get_platescale(self, file):
        if self.plate_scale is None and \
           'pltscl' in self.astrometry_parameters:
            self.plate_scale = self.astrometry_parameters.pop('pltscl', None)
        return self.plate_scale

    def get_filename(self, filegroup):
        keys = Formatter().parse(self.save_file_name)
        k = {}
        for i in keys:
            # Assume all have same value
            _, i, _, _ = i
            if i is not None:
                k[i] = filegroup.values(i, unique=True)[0]
        name = self.save_file_name.format(**k)
        return os.path.join(self.prod_dir, self.save_file_dir, name)

    def get_psi(self, filegroup):
        """Get analyzer os retarder positions in degrees."""
        raise NotImplementedError('positions not implemented for this'
                                  'pipeline')

    def get_retarder(self, filegroup):
        """The retarder identifier, used for what polarimetry will perform."""
        raise NotImplementedError('retarder not implemented for this'
                                  'pipeline')

    def get_analyzer(self, filegroup):
        """The analyzer identifier, for polarimetry function selection."""
        raise NotImplementedError('analyzer not implemented for this'
                                  'pipeline')

    def get_filter(self, filegroup):
        return filegroup.values(self.filter_key, unique=True)[0]

    def select_catalog(self, filter):
        if filter not in self.standard_catalogs.keys():
            raise ValueError('Filter {} not supported.'.format(filter))
        return default_catalogs[self.standard_catalogs[filter]]

    def validate_data(self, filegroup):
        """Validate if it's everything ok with the data."""
        if len(filegroup.summary) < 4:
            raise ValueError('Not enough calibed images to process.')
        return True

    def calculate_polarimetry(self, fg, analyzer, retarder, psi, catalog,
                              sci_catalog, astrometry, **kwargs):
        # instrument specific tasks
        raise NotImplementedError('Polarization calculation not implemented.')

        # Here, a simple example of a dual beam polarimeter
        return  dualbeam_polarimetry(fg.calibed_files, psi, positions=None,
                                     identify_catalog=catalog,
                                     retarder=retarder,
                                     analyzer=analyzer,
                                     match_pairs_tolerance=1.0,
                                     calculate_mode='sum',
                                     science_catalog=sci_catalog,
                                     **self.astrometry_kwargs,
                                     **self.photometry_kwargs,
                                     **kwargs)


    def _process(self, prod, sci_catalog=None, astrometry=True):
        if prod.calibed_files is None:
            raise ValueError("Product failed: No calibrated images. raw files:"
                             " {}".format(prod.files.files))

        fm = FileManager(self.image_ext)
        fg = fm.create_filegroup(files=prod.calibed_files,
                                 ext=self.image_ext)
        self.validate_data(fg)

        filt = self.get_filter(fg)
        cat = self.select_catalog(filt)

        analyzer = self.get_analyzer(fg)
        retarder = self.get_retarder(fg)
        psi = self.get_psi(fg)

        stacked, pol, wcs = self.calculate_polarimetry(fg, analyzer, retarder,
                                                       psi, cat, sci_catalog,
                                                       astrometry)

        pol.meta['astropop n_images'] = len(prod.calibed_files)
        pol.meta['astropop night'] = prod.files.values('night',
                                                        unique=True)[0]
        pol.meta['astropop filter'] = filt

        stacked.header['astropop n_images'] = len(prod.calibed_files)
        stacked.header['astropop night'] = prod.files.values('night',
                                                             unique=True)[0]
        stacked.header['astropop filter'] = filt

        header=stacked.header
        if wcs is not None:
            header.update(wcs.to_header(relax=True))

        imhdu = fits.PrimaryHDU(stacked.data, header=header)
        tbhdu = fits.BinTableHDU(pol, name='polarimetry')

        if prod.sci_result is None:
            filename = self.get_filename(fg)
        else:
            filename = prod.sci_result

        mkdir_p(os.path.dirname(filename))
        fits.HDUList([imhdu, tbhdu]).writeto(filename)

    def process_products(self, products, science_catalog=None,
                         astrometry=True):
        '''Process the photometry.'''
        if not check_iterable(products):
            products = [products]
        for p in products:
            try:
                self._process(p, science_catalog, astrometry)
            except Exception as e:
                # raise e
                logger.error("Product not processed due: {}: {}"
                             .format(type(e).__name__, e))
