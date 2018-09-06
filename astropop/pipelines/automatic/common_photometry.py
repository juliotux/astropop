import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from string import Formatter

from ...catalogs import default_catalogs
from ...photometry.lightcurve import temporal_photometry
from ...image_processing.register import hdu_shift_images
from ...image_processing.imarith import imarith
from ...file_manager import FileManager
from ...py_utils import check_iterable, mkdir_p
from ...fits_utils import check_hdu
from ...logger import logger
from ..photometry_scripts import process_calib_photometry


# TODO: more manual RA,DEC set, to be customized
class StackedPhotometryPipeline():
    photometry_parameters = {}
    astrometry_parameters = {}
    combine_parameters = {}
    save_file_name = '{object}_{filter}_{night}.fits'
    save_file_dir = 'stacked_photometry/'
    filter_key = 'filter'
    standard_catalogs = {'U': 'Simbad',
                         'B': 'UCAC4',
                         'V': 'UCAC4',
                         'R': 'UCAC5',
                         'I': 'DENIS'}

    def __init__(self, product_dir, image_ext=0):
        self.prod_dir = product_dir
        self.image_ext = image_ext

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

    def get_filter(self, filegroup):
        return filegroup.values(self.filter_key, unique=True)[0]

    def select_catalog(self, filter):
        if filter not in self.standard_catalogs.keys():
            raise ValueError('Filter {} not supported.'.format(filter))
        return default_catalogs[self.standard_catalogs[filter]]

    def _process(self, prod, sci_catalog, astrometry=True):
        if prod.calibed_files is None:
            raise ValueError("Product failed: No calibrated images. raw files:"
                             " {}".format(prod.files.files))

        fm = FileManager(self.image_ext)
        if check_iterable(prod.calibed_files):
            if len(prod.calibed_files) > 1:
                fg = fm.create_filegroup(files=prod.calibed_files,
                                         ext=self.image_ext)
                stacked = None
                for i in fg.hdus():
                    if stacked is None:
                        stacked = i
                    else:
                        s = hdu_shift_images([stacked, i], method='fft')[1]
                        stacked = imarith(stacked, s, '+')
            elif len(prod.calibed_files) == 1:
                stacked = prod.calibed_files[0]
                fg = fm.create_filegroup(files=prod.calibed_files,
                                         ext=self.image_ext)
                stacked = check_hdu(stacked, ext=self.image_ext)
            else:
                raise ValueError("Product failed: No calibrated images."
                                 "raw files: {}".format(prod.files.files))
                return
        else:
            stacked = prod.calibed_files
            fg = fm.create_filegroup(files=prod.calibed_files,
                                     ext=self.image_ext)
            stacked = check_hdu(stacked, ext=self.image_ext)

        filt = self.get_filter(fg)
        cat = self.select_catalog(filt)

        wcs = WCS(stacked.header, relax=True)
        if '' in wcs.wcs.ctype or astrometry:
            wcs = None

        phot, wcs = process_calib_photometry(stacked,
                                             science_catalog=sci_catalog,
                                             identify_catalog=cat,
                                             filter=filt, wcs=wcs,
                                             return_wcs=True,
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

        imhdu = fits.PrimaryHDU(stacked.data, header=header)
        tbhdu = fits.BinTableHDU(phot, name='photometry')

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


class LightCurvePipeline():
    photometry_parameters = {}
    astrometry_parameters = {}
    combine_parameters = {}
    save_file_name = '{object}_{filter}.tsv'
    save_file_dir = 'light_curve/'
    filter_key = 'filter'
    standard_catalogs = {'U': 'Simbad',
                         'B': 'APASS',
                         'V': 'APASS',
                         'R': 'GSC2.3',
                         'I': 'DENIS'}

    def __init__(self, product_dir, image_ext=0):
        self.prod_dir = product_dir
        self.image_ext = image_ext

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

    def get_filter(self, filegroup):
        return filegroup.values(self.filter_key, unique=True)[0]

    def select_catalog(self, filter):
        if filter not in self.standard_catalogs.keys():
            raise ValueError('Filter {} not supported.'.format(filter))
        return default_catalogs[self.standard_catalogs[filter]]

    def _process(self, prod):
        # TODO: need a better design to select science and ref stars
        if prod.calibed_files is None:
            raise ValueError("Product failed: No calibrated images. raw files:"
                             " {}".format(prod.files.files))

        fm = FileManager(ext=self.image_ext)
        if check_iterable(prod.calibed_files):
            if len(prod.calibed_files) < 2:
                raise ValueError('Not enought images to generate lightcurve.')

        fg = fm.create_filegroup(files=prod.calibed_files,
                                 ext=self.image_ext)

        phot = temporal_photometry(fg.files, ext=self.image_ext,
                                   **self.photometry_parameters)
        if prod.sci_result is None:
            filename = self.get_filename(fg)
        else:
            filename = prod.sci_result
        mkdir_p(os.path.dirname(filename))
        phot.write(filename, format='ascii')
        prod.sci_result = filename

    def process_products(self, products):
        '''Process the photometry.'''
        if not check_iterable(products):
            products = [products]
        for p in products:
            try:
                self._process(p)
            except Exception as e:
                # raise e
                logger.error("Product not processed due: {}: {}"
                             .format(type(e).__name__, e))
