# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import sep
import numpy as np
import numpy.testing as npt
import pytest_check as check

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.ndimage.filters import convolve

from ._utils import _sep_fix_byte_order
from ..math.moffat import moffat_r, moffat_fwhm, PSFMoffat2D
from ..math.gaussian import gaussian_r, gaussian_fwhm, PSFGaussian2D
from ..math.array import trim_array, xy2r
from ..logger import logger

from astropop.photometry import (gen_filter_kernel, background, sepfind, daofind,
                                 starfind, sources_mask, _fwhm_loop, calc_fwhm, 
                                 _recenter_loop, recenter_sources)

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_gen_filter_kernel():
    pass

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_background():
    pass

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_sepfind():
    pass

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_daofind():
    pass

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_starfind():
    pass

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_sources_mask():
    pass

# @pytest.mark.parametrize('r', [2, 3, 4])
def test__fwhm_loop():
    pass

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_calc_fwhm():
    pass

# @pytest.mark.parametrize('r', [2, 3, 4])
def test__recenter_loop():
    pass

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_recenter_sources():
    pass

