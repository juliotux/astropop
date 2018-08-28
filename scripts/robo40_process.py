#!/bin/env python3

import os
import datetime
from optparse import OptionParser

from astropop.pipelines.automatic.opd import ROBO40Calib, ROBO40Photometry
from astropop.catalogs import ASCIICatalogClass
from astropop.py_utils import mkdir_p
from astropop.logger import logger

def main():
    parser = OptionParser("usage: %prog [options] raw_dir [raw_dir2, ...]")
    parser.add_option("-s", "--stack", action="store_true", dest="stack_images",
                      default=False,
                      help="Stack all science images in one (sum)")
    parser.add_option("-i", "--individual", action="store_true",
                      dest="save_calibed",
                      default=True,
                      help="Save individual calibed science images "
                           "in 'calib_images' subfolder")
    parser.add_option("-v", "--verbose", dest="verbose",
                      action="store_true",
                      default=False,
                      help="Enable 'DEBUG' output in python log")
    parser.add_option("-a", "--astrometry", dest="astrometry",
                      action="store_true",
                      default=False,
                      help="Enable astrometry solving of stacked images "
                           "with astrometry.net")
    parser.add_option("-n", "--science-catalog", dest="science_catalog",
                      default=None, metavar="FILE",
                      help="ASCII catalog to identify science stars. "
                           "Has to be astropy's table readable with columns "
                           "ID, RA, DEC")
    parser.add_option("-l", "--save-log", dest="save_log",
                      default=None, metavar="FILE",
                      help="Save log to FILE. If '%date' value, automatic name"
                           " based on date will be created.")
    parser.add_option("-d", "--dest", dest="reduced_folder",
                      default='~/astropop_reduced', metavar="FOLDER",
                      help="Reduced images (and created calib frames) will "
                           "be saved at inside FOLDER")
    parser.add_option("-c", "--calib", dest="calib_folder",
                      default=None, metavar="FOLDER",
                      help="Load/save calibration frames from/in FOLDER. "
                           "If not set, reduced_folder/calib will be used"
                           " instead.")

    (options, args) = parser.parse_args()

    if len(args) < 1:
        raise ValueError('No raw folder passed!')

    raw_dirs = args

    if options.verbose:
        logger.setLevel('DEBUG')
    else:
        logger.setLevel('INFO')

    stack_images = options.stack_images
    individual = options.save_calibed
    astrometry = options.astrometry
    reduced_folder = os.path.expanduser(options.reduced_folder)
    reduced_folder = os.path.abspath(reduced_folder)
    mkdir_p(reduced_folder)

    if options.calib_folder is not None:
        calib_folder = os.path.expanduser(options.calib_folder)
        calib_folder = os.path.abspath(calib_folder)
    else:
        calib_folder = os.path.join(reduced_folder, 'calib')

    sci_cat = options.science_catalog
    if sci_cat is not None:
        sci_cat = ASCIICatalogClass(sci_cat, id_key='ID', ra_key='RA',
                                    dec_key='DEC', format='ascii')

    mkdir_p(reduced_folder)
    pipe = ROBO40Calib(product_dir=reduced_folder,
                       calib_dir=calib_folder,
                       ext=1, fits_extensions=['.fz'], compression=True)
    pipe_phot = ROBO40Photometry(product_dir=reduced_folder,
                                 image_ext=1)

    def _process():
        for fold in raw_dirs:
            prods = pipe.run(fold, stack_images=stack_images,
                            save_calibed=individual,
                            astrometry=astrometry)
            pipe_phot.process_products(prods, sci_cat)

    if options.save_log is not None:
        name = options.save_log
        if name == '%date':
            d = datetime.datetime.now()
            d = d.isoformat(timespec='seconds')
            name = "astropop_{}.log".format(d)
            name = os.path.join(reduced_folder, name)
        with logger.log_to_file(name):
            _process()
    else:
        _process()

if __name__ == '__main__':
    main()
