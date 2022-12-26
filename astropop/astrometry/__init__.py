# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Process astrometry calibration using several algorithms."""

from .astrometrynet import AstrometryNetUnsolvedField, \
                           AstrometrySolver, \
                           AstrometricSolution, \
                           solve_astrometry_image, \
                           solve_astrometry_xy, \
                           solve_astrometry_hdu, \
                           solve_astrometry_framedata, \
                           create_xyls
from .manual_wcs import wcs_from_coords
from .coords_utils import guess_coordinates

__all__ = ['guess_coordinates', 'wcs_from_coords', 'AstrometrySolver',
           'AstrometricSolution', 'solve_astrometry_hdu',
           'solve_astrometry_xy', 'solve_astrometry_image',
           'solve_astrometry_framedata', 'create_xyls',
           'AstrometryNetUnsolvedField']
