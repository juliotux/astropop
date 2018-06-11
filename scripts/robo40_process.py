""""""

import sys
import os
from optparse import OptionParser

try:
    from astropop.pipelines.automatic.opd import ROBO40Calib
    from astropop.py_utils import mkdir_p
    from astropop.logger import logger
except ModuleNotFoundError:
    folder = os.path.dirname(__file__)
    folder = os.path.join(folder, '..')
    folder = os.path.abspath(folder)
    sys.path.append(folder)
    from astropop.pipelines.automatic.opd import ROBO40Calib
    from astropop.py_utils import mkdir_p
    from astropop.logger import logger


def main():
    parser = OptionParser("usage: %prog [options] raw_dir [raw_dir2, ...]")
    parser.add_option("-s", "--stack", action="store_true", dest="stack_images",
                      default=False,
                      help="Stack all science images in one (sum)")
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

    stack_images = options.stack_images
    reduced_folder = os.path.expanduser(options.reduced_folder)
    reduced_folder = os.path.abspath(reduced_folder)
    mkdir_p(reduced_folder)

    if options.calib_folder is not None:
        calib_folder = os.path.expanduser(options.calib_folder)
        calib_folder = os.path.abspath(calib_folder)
    else:
        calib_folder = os.path.join(reduced_folder, 'calib')

    mkdir_p(reduced_folder)
    pipe = ROBO40Calib(product_dir=reduced_folder,
                       calib_dir=calib_folder,
                       ext=1, fits_extensions=['.fz'], compression=True)

    for fold in raw_dirs:
        prods = pipe.run(fold, stack_images=stack_images)
        print(prods)

if __name__ == '__main__':
    logger.setLevel('DEBUG')
    main()
