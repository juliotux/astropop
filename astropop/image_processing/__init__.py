# Licensed under a 3-clause BSD style license - see LICENSE.rst

# pylint: skip-all

from .ccd_processing import *
from .imarith import *
import ccdproc
import register

register_available_methods = [None, 'fft']
combine_available_methods = ['median', 'sum', 'average']
