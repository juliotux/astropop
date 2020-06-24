# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
import numpy.testing as npt
import pytest_check as check

from astropy import units as u
from astropop.image_processing.ccd_processing import cosmics_lacosmic, gain_correct, \
                                                     subtract_bias, subtract_dark, \
                                                     flat_correct
from astropop.framedata import FrameData


@pytest.mark.parametrize('inplace', [True, False])
def test_simple_flat(inplace):
    
    expect = np.ones((20, 20))*3
    expect[0:5, 0:5] = 3/0.5
    
    # Checking flat division:
    frame1 = FrameData(np.ones((20, 20))*3, unit=u.adu)
    
    master_flat_dimless = FrameData(np.ones((20, 20)), unit=None)
    master_flat_dimless.data[0:5, 0:5] = 0.5
    
    res1 = flat_correct(frame1, master_flat_dimless, inplace=inplace)
    
    check.is_true(isinstance(res1, FrameData))
    npt.assert_array_equal(res1.data, expect)
    check.equal(res1.header['hierarch astropop flat_corrected'], True)
    
    # # Checking flat-corrected frame unit:
    # check.equal(res1.unit, u.Unit('adu'))
    
    # Check inplace statement:
    if inplace:
        check.is_true(res1.data is frame1.data)
    else:
        check.is_false(res1.data is frame1.data)
    
    
@pytest.mark.parametrize('inplace', [True, False])
def test_simple_bias(inplace):
    
    expected = np.ones((20, 20))*2
    # expected[0:5, 0:5] = 1.5
    expected[0:5, 0:5] = 2.5
    
    frame4bias = FrameData(np.ones((20, 20))*3, unit='adu')
    
    master_bias = FrameData(np.ones((20,20)),unit='adu')
    master_bias.data[0:5, 0:5] = 0.5
    
    res4 = subtract_bias(frame4bias, master_bias, inplace=inplace)

    check.is_true(isinstance(res4, FrameData))
    npt.assert_array_equal(res4.data, expected)   # Assertion error
    check.equal(res4.header['hierarch astropop bias_corrected'], True)
    
    # # Checking bias-subtracted frame unit:
    # check.equal(res1.unit, u.Unit('adu'))
    
    # Check inplace statement:
    if inplace:
        check.is_true(res4.data is frame4bias.data)
    else:
        check.is_false(res4.data is frame4bias.data)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    