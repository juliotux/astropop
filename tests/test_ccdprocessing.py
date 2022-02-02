# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np

from astropy import units as u
from astropop.image.processing import cosmics_lacosmic, \
                                      gain_correct, \
                                      subtract_bias, \
                                      subtract_dark, \
                                      flat_correct
from astropop.framedata import FrameData
from astropop.testing import assert_is_instance, assert_equal, \
                             assert_is, assert_is_not


@pytest.mark.skip
class Test_Processing_Cosmics():
    pass


@pytest.mark.skip
class Test_Processing_Gain():
    pass


class Test_Processing_Flat():
    @pytest.mark.parametrize('inplace', [True, False])
    def test_simple_flat(self, inplace):
        expect = np.ones((20, 20))*3
        expect[0:5, 0:5] = 3/0.5

        # Checking flat division:
        frame1 = FrameData(np.ones((20, 20))*3, unit=u.adu)

        master_flat_dimless = FrameData(np.ones((20, 20)), unit=None)
        master_flat_dimless.data[0:5, 0:5] = 0.5

        res1 = flat_correct(frame1, master_flat_dimless, inplace=inplace)

        assert_is_instance(res1, FrameData)
        assert_equal(res1.data, expect)
        assert_equal(res1.header['hierarch astropop flat_corrected'], True)

        if inplace:
            assert_is(res1.data, frame1.data)
        else:
            assert_is_not(res1.data, frame1.data)


class Test_Processing_Bias():
    @pytest.mark.parametrize('inplace', [True, False])
    def test_simple_bias(self, inplace):
        expected = np.ones((20, 20))*2
        expected[0:5, 0:5] = 2.5

        frame4bias = FrameData(np.ones((20, 20))*3, unit='adu')

        master_bias = FrameData(np.ones((20, 20)), unit='adu')
        master_bias.data[0:5, 0:5] = 0.5

        res4 = subtract_bias(frame4bias, master_bias, inplace=inplace)

        assert_is_instance(res4, FrameData)
        assert_equal(res4.data, expected)
        assert_equal(res4.header['hierarch astropop bias_corrected'], True)

        if inplace:
            assert_is(res4.data, frame4bias.data)
        else:
            assert_is_not(res4.data, frame4bias.data)


@pytest.mark.skip
class Test_Processing_Dark():
    pass
