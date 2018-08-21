# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .aperture import aperture_photometry
from .detection import (background, sexfind, daofind, calc_fwhm,
                        recenter_sources)
from .solve_photometry import (solve_photometry_median,
                               solve_photometry_average,
                               solve_photometry_montecarlo)
from ._phot import process_photometry

psf_available_models = ['gaussian', 'moffat']
photometry_available_methods = ['aperture']
solve_photometry_available_methods = ['median', 'average', 'montecarlo']

__all__ = ['process_photometry', 'aperture_photometry', 'background',
           'sexfind', 'daofind', 'calc_fwhm', 'recenter_sources',
           'solve_photometry_median', 'solve_photometry_average',
           'solve_photometry_montecarlo', 'psf_available_models',
           'photometry_available_methods',
           'solve_photometry_available_methods']
