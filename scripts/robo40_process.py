""""""

import sys
import os

try:
    from astropop.pipelines.automatic.opd import ROBO40Calib
    from astropop.py_utils import mkdir_p
    from astropop.logger import logger
except ModuleNotFoundError:
    sys.path.append('..')
    from astropop.pipelines.automatic.opd import ROBO40Calib
    from astropop.py_utils import mkdir_p
    from astropop.logger import logger


def main(args):
    if len(args) == 1:
        raise ValueError("usage: python robo40_process.py raw_folder [product_folder]")
    fold = args[1]
    while fold[-1] == '/':
        fold = fold[:-1]
    last_name = os.path.split(fold)[1]

    if len(args) > 2:
        red_fold = args[2]
    else:
        red_fold = os.path.join(os.path.expanduser('~/astropop'), last_name)

    mkdir_p(red_fold)

    pipe = ROBO40Calib(product_dir=red_fold,
                       ext=1, fits_extensions=['.fz'], compression=True)

    pipe.run(fold, stack_images=False)

if __name__ == '__main__':
    logger.setLevel('DEBUG')
    main(sys.argv)
