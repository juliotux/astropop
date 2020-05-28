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
    frame1 = FrameData(np.ones((20, 20))*3, unit='adu')
    master_flat = FrameData(np.ones((20, 20)), unit=None)
    master_flat.data[0:5, 0:5] = 0.5
    expect = np.ones((20, 20))
    expect *= 3
    expect[0:5, 0:5] = 3/0.5

    res = flat_correct(frame1, master_flat, inplace=inplace)

    check.is_true(isinstance(res, FrameData))
    npt.assert_array_equal(res.data, expect)
    check.equal(res.header['hierarch astropop flat_corrected'], True)
    check.equal(res.unit, u.Unit('adu'))

    if inplace:
        check.is_true(res is frame 1)
    else:
        check.is_false(res is frame 1)
        
