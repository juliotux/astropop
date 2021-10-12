# Licensed under a 3-clause BSD style license - see LICENSE.rst

# from .calibration import ()
from .imarith import imarith  # noqa: F401
from .imcombine import imcombine, ImCombiner  # noqa: F401
from . import register  # noqa: F401

register_available_methods = [None, 'fft']
combine_available_methods = ['median', 'sum', 'average']
