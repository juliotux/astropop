# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .aperture import aperture_photometry

psf_available_models = ['gaussian', 'moffat']
photometry_available_methods = ['aperture']
solve_photometry_available_methods = ['median', 'average', 'montecarlo']


__all__ = ['aperture_photometry']
