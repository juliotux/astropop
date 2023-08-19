# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
from astropop.image.processing import cosmics_lacosmic, \
                                      gain_correct, \
                                      subtract_bias, \
                                      subtract_dark, \
                                      flat_correct, \
                                      trim_image
from astropop.framedata import FrameData, PixelMaskFlags
from astropop.math import QFloat
from astropop.testing import *
from astropy import units as u
from astropy.wcs import WCS
import numpy as np


class TestProcessingCosmics:
    @pytest.mark.parametrize('inplace', [True, False])
    def test_lacosmic_numpy(self, inplace):
        arr = np.ones((10, 10))
        arr[3, 3] = 100000
        ccd = cosmics_lacosmic(arr, inplace=inplace)
        assert_is_instance(ccd, FrameData)
        assert_equal(ccd.data, np.ones((10, 10)))
        assert_true(ccd.header['astropop lacosmic'])

    @pytest.mark.parametrize('inplace', [True, False])
    def test_lacosmic_qfloat(self, inplace):
        arr = np.ones((10, 10))
        arr[3, 3] = 100000
        ccd = cosmics_lacosmic(QFloat(arr, unit='adu'), inplace=inplace)
        assert_is_instance(ccd, FrameData)
        assert_equal(ccd.data, np.ones((10, 10)))
        assert_true(ccd.header['astropop lacosmic'])

    @pytest.mark.parametrize('inplace', [True, False])
    def test_lacosmic_framedata(self, inplace):
        arr = np.ones((10, 10))
        arr[3, 3] = 100000
        frame = FrameData(arr, unit='adu')
        ccd = cosmics_lacosmic(frame, inplace=inplace)
        assert_is_instance(ccd, FrameData)
        assert_equal(ccd.data, np.ones((10, 10)))
        assert_true(ccd.header['astropop lacosmic'])
        if inplace:
            assert_is(ccd, frame)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_lacosmic_mask(self, inplace):
        arr = np.ones((10, 10))
        arr[3, 3] = 100000
        mask = np.zeros((10, 10), dtype='uint8')
        mask[3, 3] |= PixelMaskFlags.COSMIC_RAY.value
        mask[3, 3] |= PixelMaskFlags.INTERPOLATED.value
        ccd = cosmics_lacosmic(arr, inplace=inplace)
        assert_is_instance(ccd, FrameData)
        assert_equal(ccd.data, np.ones((10, 10)))
        assert_equal(ccd.flags, mask)
        assert_equal(ccd.mask, np.zeros((10, 10), dtype=bool))
        assert_true(ccd.header['astropop lacosmic'])


class TestProcessingGain:
    @pytest.mark.parametrize('inplace', [True, False])
    def test_gain_value(self, inplace):
        gain = 2.0
        f = FrameData(np.ones((10, 10)), unit='adu')
        ccd = gain_correct(f, gain, inplace)
        assert_is_instance(ccd, FrameData)
        assert_almost_equal(ccd.data, np.ones((10, 10))*2)
        assert_equal(ccd.unit, 'adu')
        assert_true(ccd.header['astropop gain_corrected'])
        assert_equal(ccd.header['astropop gain_corrected_value'], 2.0)
        assert_equal(ccd.header['astropop gain_corrected_unit'], '')
        if inplace:
            assert_is(f, ccd)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_gain_qfloat(self, inplace):
        gain = QFloat(2.0, unit='electron/adu')
        f = FrameData(np.ones((10, 10)), unit='adu')
        ccd = gain_correct(f, gain, inplace)
        assert_is_instance(ccd, FrameData)
        assert_almost_equal(ccd.data, np.ones((10, 10))*2)
        assert_equal(ccd.unit, 'electron')
        assert_true(ccd.header['astropop gain_corrected'])
        assert_equal(ccd.header['astropop gain_corrected_value'], 2.0)
        assert_equal(ccd.header['astropop gain_corrected_unit'],
                     u.Unit('electron/adu'))
        if inplace:
            assert_is(f, ccd)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_gain_quantity(self, inplace):
        gain = 2.0*u.Unit('electron/adu')
        f = FrameData(np.ones((10, 10)), unit='adu')
        ccd = gain_correct(f, gain, inplace)
        assert_is_instance(ccd, FrameData)
        assert_almost_equal(ccd.data, np.ones((10, 10))*2)
        assert_equal(ccd.unit, 'electron')
        assert_true(ccd.header['astropop gain_corrected'])
        assert_equal(ccd.header['astropop gain_corrected_value'], 2.0)
        assert_equal(ccd.header['astropop gain_corrected_unit'],
                     u.Unit('electron/adu'))
        if inplace:
            assert_is(f, ccd)


class TestProcessingTrimImage:
    @pytest.mark.parametrize('inplace', [True, False])
    def test_trim_x(self, inplace):
        yi, xi = np.indices((100, 100))
        xi = FrameData(xi)
        trimmed = trim_image(xi, slice(20, 30, 1), None, inplace)
        assert_is_instance(trimmed, FrameData)
        assert_equal(trimmed.shape, (100, 10))
        assert_equal(trimmed.data, [list(range(20, 30, 1))]*100)
        assert_equal(trimmed.header['astropop trimmed_section'],
                     '20:30,0:100')
        if inplace:
            assert_is(trimmed, xi)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_trim_y(self, inplace):
        yi, xi = np.indices((100, 100))
        yi = FrameData(yi)
        trimmed = trim_image(yi, None, slice(20, 30, 1), inplace)
        assert_is_instance(trimmed, FrameData)
        assert_equal(trimmed.shape, (10, 100))
        assert_equal(trimmed.data, [[i]*100 for i in range(20, 30, 1)])
        assert_equal(trimmed.header['astropop trimmed_section'],
                     '0:100,20:30')
        if inplace:
            assert_is(trimmed, yi)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_trim_xy(self, inplace):
        arr = np.arange(100*100).reshape((100, 100))
        frame = FrameData(arr)
        trimmed = trim_image(frame, slice(40, 50, 1), slice(20, 30, 1), inplace)
        assert_is_instance(trimmed, FrameData)
        assert_equal(trimmed.shape, (10, 10))
        assert_equal(trimmed.data, arr[20:30, 40:50])
        assert_equal(trimmed.header['astropop trimmed_section'],
                     '40:50,20:30')
        if inplace:
            assert_is(trimmed, frame)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_trim_nparray(self, inplace):
        arr = np.arange(100*100).reshape((100, 100))
        trimmed = trim_image(arr, slice(40, 50, 1), slice(20, 30, 1), inplace)
        assert_is_instance(trimmed, FrameData)
        assert_equal(trimmed.shape, (10, 10))
        assert_equal(trimmed.data, arr[20:30, 40:50])
        assert_equal(trimmed.header['astropop trimmed_section'],
                     '40:50,20:30')
        assert_is_not(trimmed, arr)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_trim_with_wcs(self, inplace):
        arr = np.arange(100*100).reshape((100, 100))
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [50, 50]
        wcs.wcs.cdelt = [0.5, 0.5]
        wcs.wcs.crval = [15.5, 20.5]
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        frame = FrameData(arr, wcs=wcs)
        trimmed = trim_image(frame, slice(40, 50, 1), slice(20, 30, 1),
                             inplace)
        assert_is_instance(trimmed, FrameData)
        assert_equal(trimmed.shape, (10, 10))
        assert_equal(trimmed.data, arr[20:30, 40:50])
        assert_equal(trimmed.header['astropop trimmed_section'],
                     '40:50,20:30')
        assert_equal(trimmed.wcs.wcs.crpix, [10, 30])
        assert_equal(trimmed.wcs.wcs.cdelt, [0.5, 0.5])
        assert_equal(trimmed.wcs.wcs.crval, [15.5, 20.5])
        assert_equal(trimmed.wcs.wcs.ctype, ['RA---TAN', 'DEC--TAN'])
        if inplace:
            assert_is(trimmed, frame)
