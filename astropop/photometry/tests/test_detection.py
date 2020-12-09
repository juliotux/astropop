# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import sep
import numpy as np
import numpy.testing as npt
import pytest_check as check

from astropop.photometry import (background, sepfind, daofind, starfind, 
                                 calc_fwhm, recenter_sources)
from astropop.photometry.tests import (gen_image)

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_gen_filter_kernel():
    pass

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_background():
    
    # im_e = np.random.poisson(im_e) criar uma distribuição de erro poissonico 
    # Deve criar  um erro poissonico sobre uma distribuição gaussiana

    scale = 1.0
    psi = 90
    p = 0.1
    theta = 15
    
    image_test = gen_image(scale, psi, p, theta)
    
    image_cols = len(image_test)
    image_rows = len(image_test[0])
    
    box_size = image_cols
    filter_size = image_cols

    bkg_1_global = background(image_test, box_size, filter_size, mask=None, global_bkg=True)

    npt.assert_array_almost_equal(bkg_1_global[0], 298, decimal=0)
    npt.assert_array_almost_equal(bkg_1_global[1], 24.9, decimal=1)


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

