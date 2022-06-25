# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np

from astropy import units as u
from astropy.wcs import WCS
from astropop.image.processing import cosmics_lacosmic, \
                                      gain_correct, \
                                      subtract_bias, \
                                      subtract_dark, \
                                      flat_correct, \
                                      trim_image
from astropop.framedata import FrameData
from astropop.testing import *


class Test_Processing_Cosmics():
    @pytest.mark.parametrize('inplace', [True, False])
    def test_cosmics_lacosmic(self, inplace):
        image = FrameData(np.ones((20, 20))*3, unit=u.adu,
                          mask=np.zeros((20, 20)))
        # Add cosmics
        image.data[10, 10] = 35000
        image.data[10, 11] = 35000
        image.data[11, 10] = 35000
        image.mask[15, 18] = 1

        expect_mask = np.zeros((20, 20))
        # currently we are not updating the mask
        # expect_mask[10, 10] = 1
        # expect_mask[10, 11] = 1
        # expect_mask[11, 10] = 1
        expect_mask[15, 18] = 1

        # Run the cosmics removal
        res = cosmics_lacosmic(image, inplace=inplace)
        assert_equal(res.data, np.ones((20, 20))*3)
        assert_equal(res.mask, expect_mask)
        assert_equal(res.meta['astropop lacosmic'], True)

        if inplace:
            assert_is(res, image)
        else:
            assert_is_not(res, image)


class Test_Processing_Gain():
    @pytest.mark.parametrize('inplace', [True, False])
    def test_simple_gain_correct(self, inplace):
        image = FrameData(np.ones((20, 20))*3, unit=u.adu,
                          uncertainty=5, mask=np.zeros((20, 20)))
        gain = 1.5*u.Unit('electron')/u.adu
        res = gain_correct(image, gain, inplace=inplace)

        assert_equal(res.data, np.ones((20, 20))*3*1.5)
        assert_equal(res.uncertainty, np.ones((20, 20))*5*1.5)
        assert_equal(res.unit, u.Unit('electron'))

        assert_equal(res.meta['astropop gain_corrected'], True)
        assert_equal(res.meta['astropop gain_corrected_value'], 1.5)
        assert_equal(res.meta['astropop gain_corrected_unit'],
                     'electron / adu')

        if inplace:
            assert_is(res, image)
        else:
            assert_is_not(res, image)


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
        assert_equal(res1.header['astropop flat_corrected'], True)

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
        assert_equal(res4.header['astropop bias_corrected'], True)

        if inplace:
            assert_is(res4.data, frame4bias.data)
        else:
            assert_is_not(res4.data, frame4bias.data)


@pytest.mark.skip
class Test_Processing_Dark():
    pass


class Test_Processing_TrimImage():
    @pytest.mark.parametrize('inplace', [True, False])
    def test_simple_trim(self, inplace):
        shape = (512, 1024)
        arr = np.random.uniform(20, 30, shape)
        mask = np.zeros(shape)
        mask[0:10, 0:40] = 1
        image = FrameData(arr, unit=u.adu, uncertainty=np.sqrt(arr),
                          mask=mask)

        # Trim the image
        xslice = slice(5, 25)
        yslice = slice(34, 46)
        res = trim_image(image, xslice, yslice, inplace=inplace)

        section = (yslice, xslice)

        assert_almost_equal(res.data, arr[section])
        assert_almost_equal(res.uncertainty, np.sqrt(arr)[section])
        assert_almost_equal(res.mask, mask[section])
        assert_equal(res.header['astropop trimmed_section'], '5:25,34:46')

        if inplace:
            assert_is(res, image)
        else:
            assert_is_not(res, image)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_wcs_trim(self, inplace):
        shape = (512, 1024)
        arr = np.random.uniform(20, 30, shape)
        mask = np.zeros(shape)
        mask[0:10, 0:40] = 1
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [100, 100]
        wcs.wcs.cdelt = [1, 1]
        wcs.wcs.crval = [0, 0]
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.pc = [[1, 0], [0, 1]]

        image = FrameData(arr, unit=u.adu, uncertainty=np.sqrt(arr),
                          mask=mask, wcs=wcs)

        # Trim the image
        xslice = slice(5, 25)
        yslice = slice(34, 46)
        res = trim_image(image, xslice, yslice, inplace=inplace)

        section = (yslice, xslice)

        assert_almost_equal(res.data, arr[section])
        assert_almost_equal(res.uncertainty, np.sqrt(arr)[section])
        assert_almost_equal(res.mask, mask[section])
        assert_equal(res.header['astropop trimmed_section'], '5:25,34:46')
        assert_equal(res.wcs.wcs.crpix, [100-5, 100-34])
        assert_equal(res.wcs.wcs.cdelt, [1, 1])
        assert_equal(res.wcs.wcs.crval, [0, 0])
        assert_equal(res.wcs.wcs.ctype, ['RA---TAN', 'DEC--TAN'])
        assert_equal(res.wcs.wcs.pc, [[1, 0], [0, 1]])

        if inplace:
            assert_is(res, image)
        else:
            assert_is_not(res, image)
