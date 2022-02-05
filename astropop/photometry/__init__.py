# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .aperture import aperture_photometry
from .detection import (background, sepfind, daofind, calc_fwhm,
                        recenter_sources, starfind, gen_filter_kernel,
                        sources_mask)
from .solve_photometry import (solve_photometry_median,
                               solve_photometry_average,
                               solve_photometry_montecarlo)

psf_available_models = ['gaussian', 'moffat']
photometry_available_methods = ['aperture']
solve_photometry_available_methods = ['median', 'average', 'montecarlo']

__all__ = ['aperture_photometry', 'background', 'sepfind', 'daofind',
           'calc_fwhm', 'recenter_sources', 'starfind',
           'solve_photometry_median', 'solve_photometry_average',
           'solve_photometry_montecarlo', 'gen_filter_kernel',
           'sources_mask']
