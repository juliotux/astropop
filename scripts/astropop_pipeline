#!/bin/env python3

import os
import datetime
from optparse import OptionParser

from astropop.pipelines.config_based import MasterPipeline
from astropop.py_utils import mkdir_p
from astropop.logger import logger

import numpy as np
np.warnings.filterwarnings('ignore')

def main():
    parser = OptionParser("usage: %prog [options] config_file.json "
                          "[config_file2.json, ...]"
                          "\n\nAtention:\ncalib, raw and product folder must"
                          "be set in the config files!"
                          "\n\nUse -vvv for DEBUG.")
    parser.add_option("-v", "--verbose", dest="verbose",
                      action="count",
                      help="Enable 'DEBUG' output in python log")
    parser.add_option("-e", "--skip-exist", dest="check_exist",
                      action="store_true",
                      default=False,
                      help="Skip if the product file already exists.")
    parser.add_option("-r", "--raise-error", dest="raise_error",
                      action="store_true",
                      default=False,
                      help="Raise errors instead of just log them.")
    parser.add_option("-i", "--include", dest="include_file",
                      default=None, metavar="FILE",
                      help="Preload FILE config before run the code.")
    parser.add_option("-p", "--product", dest="product",
                      default=None, metavar="NAME",
                      help="Product ID inside the configure files.")
    parser.add_option("-l", "--save-log", dest="save_log",
                      default=None, metavar="FILE",
                      help="Save log to FILE. If '%date' value, automatic name"
                           " based on date will be created.")

    (options, args) = parser.parse_args()

    if len(args) < 1:
        raise ValueError('No raw folder passed!')

    files = args

    if options.verbose is None:
        logger.setLevel('WARN')
    elif options.verbose == 1:
        logger.setLevel('INFO')
    else:
        logger.setLevel('DEBUG')

    pipe = MasterPipeline()
    if options.include_file is not None:
        pipe.load_default_file(options.include_file)

    def _process():
        for f in files:
            if options.product is not None:
                pipe.process_product(f, dataset=options.product,
                                     check_exist=options.check_exist,
                                     raise_error=options.raise_error)
            else:
                pipe.process_product(f, check_exist=options.check_exist,
                                     raise_error=options.raise_error)

    if options.save_log is not None:
        name = options.save_log
        if name == '%date':
            d = datetime.datetime.now()
            d = d.isoformat(timespec='seconds')
            name = "astropop_{}.log".format(d)
        with logger.log_to_file(name):
            _process()
    else:
        _process()

if __name__ == '__main__':
    main()
