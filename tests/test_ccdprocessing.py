# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np

from astropy import units as u
from astropop.image_processing.ccd_processing import cosmics_lacosmic, \
                                                     gain_correct, \
                                                     subtract_bias, \
                                                     subtract_dark, \
                                                     flat_correct
from astropop.framedata import FrameData


pytestmark = pytest.mark.skip

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
    expected[0:5, 0:5] = 2.5

    frame4bias = FrameData(np.ones((20, 20))*3, unit='adu')

    master_bias = FrameData(np.ones((20, 20)), unit='adu')
    master_bias.data[0:5, 0:5] = 0.5

    res4 = subtract_bias(frame4bias, master_bias, inplace=inplace)

    check.is_true(isinstance(res4, FrameData))
    npt.assert_array_equal(res4.data, expected)
    check.equal(res4.header['hierarch astropop bias_corrected'], True)

    # # Checking bias-subtracted frame unit:
    # check.equal(res1.unit, u.Unit('adu'))

    # Check inplace statement:
    if inplace:
        check.is_true(res4.data is frame4bias.data)
    else:
        check.is_false(res4.data is frame4bias.data)


@pytest.mark.parametrize('inplace', [True, False])
def test_simple_dark(inplace):
    frame4dark = FrameData(np.ones((20, 20))*3, unit='adu')

    master_dark0 = FrameData(np.zeros((20, 20)), unit='adu')
    master_dark0.data[0:5, 0:4] = 0.5
    master_dark1 = FrameData(np.ones((20, 20)), unit=None)
    master_dark1.data[0:5, 0:4] = 1.5
    master_dark5 = FrameData(np.ones((20, 20))*5, unit='adu')
    master_dark5.data[0:5, 0:4] = 5.5

    exposure1 = 1.0
    exposure2 = 1.2

    res5_zeros = subtract_dark(frame4dark, master_dark0, exposure1, exposure1,
                               inplace=inplace)
    res5_ones = subtract_dark(frame4dark, master_dark1, exposure1, exposure1,
                              inplace=inplace)
    res5_fives = subtract_dark(frame4dark, master_dark5, exposure1, exposure1,
                               inplace=inplace)
    res5_scaled = subtract_dark(frame4dark, master_dark1, exposure1, exposure2,
                                inplace=inplace)

    exptd4zeros = np.ones((20, 20))*(3-0)
    exptd4zeros[0:5, 0:4] = 3.0 - 0.5
    exptd4ones = np.ones((20, 20))*(3-1)
    exptd4ones[0:5, 0:4] = 3.0 - 1.5
    exptd4fives = np.ones((20, 20))*(3-5)
    exptd4fives[0:5, 0:4] = 3.0 - 5.5

    exptd4scaled = np.ones((20, 20))*(3-1*exposure2/exposure1)
    exptd4scaled[0:5, 0:4] = 3.0 - 1.5*exposure2/exposure1

    # Checking with the Ones matrix: ----------------------------------------
    check.is_true(isinstance(res5_ones, FrameData))
    npt.assert_array_equal(res5_ones.data, exptd4ones)
    check.equal(res5_ones.header['hierarch astropop dark_corrected'], True)
    check.equal(res5_ones.header['hierarch astropop dark_corrected_scale'],
                1.0)

    # # Checking dark-subtracted frame unit:
    # check.equal(res5_ones.unit, u.Unit('adu'))

    # Check inplace statement:
    if inplace:
        check.is_true(res5_ones.data is frame4dark.data)
    else:
        check.is_false(res5_ones.data is frame4dark.data)

    # Checking with the Zeros matrix
    check.is_true(isinstance(res5_zeros, FrameData))
    npt.assert_array_equal(res5_zeros.data, exptd4zeros)
    check.equal(res5_zeros.header['hierarch astropop dark_corrected'], True)
    check.equal(res5_zeros.header['hierarch astropop dark_corrected_scale'],
                1.0)

    # # Checking dark-subtracted frame unit:
    # check.equal(res5_zeros.unit, u.Unit('adu'))

    # Checking with the Fives matrix (master dark masking the signal)
    check.is_true(isinstance(res5_fives, FrameData))
    npt.assert_array_equal(res5_fives.data, exptd4fives)
    check.equal(res5_fives.header['hierarch astropop dark_corrected'], True)
    check.equal(res5_fives.header['hierarch astropop dark_corrected_scale'],
                1.0)

    # # Checking dark-subtracted frame unit:
    # check.equal(res5_fives.unit, u.Unit('adu'))

    # # Checking with Scaled Exposure
    # check.is_true(isinstance(res5_scaled, FrameData))
    # npt.assert_array_equal(res5_scaled.data, exptd4scaled)
    # check.equal(res5_scaled.header['hierarch astropop dark_corrected'], True)
    # check.equal(res5_scaled.header['hierarch astropop dark_corrected_scale'],
    #             exposure2/exposure1)
