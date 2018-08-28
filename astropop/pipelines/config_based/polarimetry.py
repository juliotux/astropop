# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import numpy as np
from collections import OrderedDict
from astropy.io import fits
from astropy.table import Table

from .base import ReducePipeline
from .photometry import PhotometryPipeline
from ...catalogs import default_catalogs, ASCIICatalogClass
from ..calib_scripts import calib_science
from ...fits_utils import check_hdu
from ...py_utils import process_list, check_iterable, mkdir_p
from ...image_processing import combine
from ...logger import logger
from ..polarimetry_scripts import process_polarimetry, run_pccdpack


phot_parameters = PhotometryPipeline().parameters


class PolarimetryPipeline(ReducePipeline):
    def __init__(self, config=None):
        super(PolarimetryPipeline, self).__init__(config=config)

    @property
    def parameters(self):
        phot_parameters.update(OrderedDict(
            retarder_key="Polarimeter retarder key position",
            retarder_type="Type of retarder: 'half' or 'quarter'",
            retarder_rotation="Retarder rotation of each position, in degrees",
            retarder_direction="Direction of the retarder rotation: "
                               "1 (following equatorial PA convention), "
                               "-1 (counter equatorial PA convention)",
            match_pairs_tolerance="Maximum tolerance to match ordinary/"
                                  "extraordinary pairs",
            delta_x="Default distance between the beans in x axis (pixels)",
            delta_y="Default distance between the beans in y axis (pixels)"
        ))
        return phot_parameters

    def run(self, name, **config):
        """Run this pipeline script"""
        product_dir = config['product_dir']
        night = config['night']
        s = [os.path.join(config['raw_dir'], i) for i in config['sources']]

        check_exist = config.get('check_exist', False)

        astropop_prod = os.path.join(product_dir, "{}_polarimetry_astropop_{}"
                                    .format(night, name))
        pccd_prod = os.path.join(product_dir, "{}_polarimetry_pccdpack_{}"
                                 .format(night, name))
        if check_exist:
            process_astropop = (config.get('astropop_pol', True) and not
                               os.path.isfile(astropop_prod))
            process_pccd = (config.get('pccdpack', False) and not
                            os.path.isfile(pccd_prod))
        else:
            process_pccd = config.get('pccdpack', False)
            process_astropop = config.get('astropop_pol', True)

        if not process_pccd and not process_astropop:
            return

        calib_kwargs = {}
        for i in ('master_bias', 'master_flat', 'dark_frame', 'badpixmask',
                  'prebin', 'gain_key', 'gain', 'rdnoise_key',
                  'combine_method', 'combine_sigma', 'exposure_key',
                  'mem_limit', 'combine_align_method',
                  'calib_dir', 'product_dir', 'remove_cosmics',
                  'save_calib_path'):
            if i in config.keys():
                calib_kwargs[i] = config[i]

        if config.get('astropop_cal', True) and (process_astropop or
                                                process_pccd):
            ccds = calib_science(s, **calib_kwargs)
        else:
            ccds = process_list(check_hdu, s)

        polkwargs = {}
        for i in ['ra_key', 'dec_key', 'gain_key', 'rdnoise_key',
                  'retarder_key', 'retarder_type', 'retarder_direction',
                  'filter', 'plate_scale', 'photometry_type',
                  'psf_model', 'r', 'r_in', 'r_out', 'psf_niters',
                  'box_size', 'detect_fwhm', 'detect_snr', 'remove_cosmics',
                  'align_images', 'solve_photometry_type',
                  'match_pairs_tolerance', 'montecarlo_iters',
                  'montecarlo_percentage', 'astrometry_calib',
                  'delta_x', 'delta_y', 'brightest_star_dec',
                  'brightest_star_ra', 'image_flip', 'image_north_direction',
                  'limit_angle', 'calculate_mode']:
            if i in config.keys():
                polkwargs[i] = config[i]

        if "identify_catalog_name" in config.keys():
            try:
                ref_cat = default_catalogs[config["identify_catalog_name"]]
                polkwargs['identify_catalog'] = ref_cat
            except KeyError:
                raise ValueError("Catalog {} not available. Available catalogs:"
                                 " {}".format(config["identify_catalog_name"],
                                              default_catalogs))

        mkdir_p(product_dir)

        image = combine(ccds, method='sum', mem_limit=config.get('mem_limit',
                                                                 1e9))

        if "science_catalog" in config.keys():
            sci_cat = ASCIICatalogClass(config['science_catalog'],
                                        id_key=config['science_id_key'],
                                        ra_key=config['science_ra_key'],
                                        dec_key=config['science_dec_key'],
                                        format=config['science_format'])
            polkwargs['science_catalog'] = sci_cat

        if process_astropop:
            if not config.get('astropop_cal', True):
                if 'save_calib_path' in config.keys():
                    ccds = [os.path.join(config['save_calib_path'],
                                         os.path.basename(i))
                            for i in s]
                    ccds = process_list(check_hdu, ccds)
            logger.info('Processing polarimetry with astropop.')
            logger.debug('Processing {} images'.format(len(ccds)))
            t, wcs, ret = process_polarimetry(ccds, **polkwargs)
            config['retarder_positions'] = ret

            hdus = []
            header_keys = ['retarder_type', 'retarder_rotation',
                           'retarder_direction', 'retarder_positions',
                           'align_images', 'solve_photometry_type',
                           'plate_scale', 'filter', 'night']
            header_keys += ['r', 'r_in', 'r_out', 'detect_fwhm',
                            'detect_snr']
            header_keys += ['psf_model', 'box_size', 'psf_niters']

            if config.get('solve_photometry_type', None) == 'montecarlo':
                header_keys += ['montecarlo_iters', 'montecarlo_percentage']

            if config.get('identify_catalog_name', None) is not None:
                header_keys += ['identify_catalog_name',
                                'identify_limit_angle']

            analyzer_key = config.get('analyzer_key')
            if analyzer_key:
                config['analyzer'] = image.header.get(analyzer_key)
                header_keys += ['analyzer']

            hdu = fits.BinTableHDU(t, name="log_table".format(i))
            for k in header_keys:
                if k in config.keys():
                    v = config[k]
                    key = 'hierarch astropop {}'.format(k)
                    if check_iterable(v):
                        hdu.header[key] = ','.join([str(m) for m in v])
                    else:
                        hdu.header[key] = v
            hdus.append(hdu)

            out = Table(dtype=t.dtype)
            for group in t.group_by('star_index').groups:
                m = np.argmax(group['p']/group['p_error'])
                out.add_row(group[m])
            out['snr'] = out['p']/out['p_error']

            hdu = fits.BinTableHDU(out, name="out_table")
            for k in header_keys:
                if k in config.keys():
                    v = config[k]
                    key = 'hierarch astropop {}'.format(k)
                    if check_iterable(v):
                        hdu.header[key] = ','.join([str(m) for m in v])
                    else:
                        hdu.header[key] = v
            hdus.append(hdu)

            if wcs is not None:
                image.header.update(wcs.to_header(relax=True))
            hdulist = fits.HDUList([image, *hdus])
            hdulist.writeto(astropop_prod, overwrite=True)

        if process_pccd:
            if not config.get('astropop_cal', True):
                if 'save_calib_path' in config.keys():
                    ccds = [os.path.join(config['save_calib_path'],
                                         os.path.basename(i))
                            for i in s]
                    ccds = process_list(check_hdu, ccds)
            logger.info('Processing polarimetry with pccdpack.')
            pccd = run_pccdpack(ccds, wcs=wcs, **polkwargs)
            wcs = pccd[3]
            hdus = []
            hdus.append(fits.BinTableHDU(pccd[0], name='out_table'))
            hdus.append(fits.BinTableHDU(pccd[1], name='dat_table'))
            hdus.append(fits.BinTableHDU(pccd[2], name='log_table'))
            header_keys = ['retarder_type', 'retarder_rotation',
                           'retarder_direction', 'retarder_positions',
                           'align_images', 'solve_photometry_type',
                           'plate_scale', 'filter', 'night',
                           'r', 'r_in', 'r_out']
            if config.get('identify_catalog_name', None) is not None:
                header_keys += ['identify_catalog_name',
                                'identify_limit_angle']

            analyzer_key = config.get('analyzer_key')
            if analyzer_key:
                config['analyzer'] = image.header.get(analyzer_key)
                header_keys += ['analyzer']

            for hdu in hdus:
                for k in header_keys:
                    if k in config.keys():
                        v = config[k]
                        key = 'hierarch astropop {}'.format(k)
                        if check_iterable(v):
                            hdu.header[key] = ','.join([str(m) for m in v])
                        else:
                            hdu.header[key] = v
            hdulist = fits.HDUList([image, *hdus])
            hdulist.writeto(pccd_prod, overwrite=True)
