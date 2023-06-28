# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Image proccessing library."""

from .imarith import imarith
from .imcombine import imcombine, ImCombiner
from .processing import (cosmics_lacosmic, gain_correct, subtract_bias,
                         subtract_dark, flat_correct, trim_image)


__all__ = ['imarith', 'imcombine', 'ImCombiner', 'cosmics_lacosmic',
           'gain_correct', 'subtract_bias', 'subtract_dark', 'flat_correct',
           'trim_image']
