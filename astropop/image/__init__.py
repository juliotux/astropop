# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .calibration import *
from .imarith import *
from . import register

register_available_methods = [None, 'fft']
combine_available_methods = ['median', 'sum', 'average']
