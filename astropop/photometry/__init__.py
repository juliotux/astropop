# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .photometry import *

psf_available_model = None
photometry_available_methods = ['aperture', 'psf']
solve_photometry_available_methods = ['median', 'average', 'montecarlo']
