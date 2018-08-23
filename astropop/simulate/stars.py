"""A simple and faster simulating engine for star fields."""
import numpy as np

from ..math.models.gaussian import gaussian_2d, gaussian_fwhm
from ..math.models.moffat import moffat_2d, moffat_fwhm


def fwhm_to_alpha(fwhm, beta):
    """Compute moffat alpha from fwhm and beta."""
    return fwhm/2*np.sqrt(2**(1/beta) - 1)
