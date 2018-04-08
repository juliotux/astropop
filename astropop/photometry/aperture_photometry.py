# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..astrometry import solve_astrometry
from ..catalogs import identify_stars
from ..py_utils import check_iterable
from ..logger import logger

try:
    from . import photutils_wrapper as phot
    _use_phot = True
except ModuleNotFoundError:
    _use_phot = False
    logger.warn('Photutils not found, ignoring it.')

try:
    from . import sep_wrapper as sep
    _use_sep = True
except ModuleNotFoundError:
    _use_sep = False
    logger.warn('SEP not found, ignoring it')
