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


class Test_Processing_Cosmics:
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
        else:
            assert_is_not(ccd, frame)

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

    @pytest.mark.parametrize('inplace', [True, False])
    def test_cosmics_lacosmic_full(self, inplace):
        image = FrameData(np.ones((20, 20))*3, unit=u.adu,
                          mask=np.zeros((20, 20)))
        # Add cosmics
        image.data[10, 10] = 35000
        image.data[10, 11] = 35000
        image.data[11, 10] = 35000
        image.flags[15, 18] = PixelMaskFlags.MASKED.value

        expect_flags = np.zeros((20, 20))
        cosmic_val = (PixelMaskFlags.COSMIC_RAY |
                      PixelMaskFlags.INTERPOLATED).value
        expect_flags[10, 10] = cosmic_val
        expect_flags[10, 11] = cosmic_val
        expect_flags[11, 10] = cosmic_val
        expect_flags[15, 18] = PixelMaskFlags.REMOVED.value

        # Run the cosmics removal
        res = cosmics_lacosmic(image, inplace=inplace)
        assert_equal(res.data, np.ones((20, 20))*3)
        assert_equal(res.flags, expect_flags)
        assert_equal(res.meta['astropop lacosmic'], True)

        if inplace:
            assert_is(res, image)
        else:
            assert_is_not(res, image)


class Test_Processing_Gain:
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
        else:
            assert_is_not(f, ccd)

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
        else:
            assert_is_not(f, ccd)

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
        else:
            assert_is_not(f, ccd)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_simple_gain_correct(self, inplace):
        image = FrameData(np.ones((20, 20))*3, unit=u.adu,
                          uncertainty=5, mask=np.zeros((20, 20)))
        gain = 1.5*u.Unit('electron')/u.adu
        res = gain_correct(image, gain, inplace=inplace)

        assert_equal(res.data, np.ones((20, 20))*3*1.5)
        assert_equal(res.uncertainty, np.ones((20, 20))*5*1.5)
        assert_equal(res.unit, u.Unit('electron'))
        assert_equal(res.flags, np.zeros((20, 20)))

        assert_equal(res.meta['astropop gain_corrected'], True)
        assert_equal(res.meta['astropop gain_corrected_value'], 1.5)
        assert_equal(res.meta['astropop gain_corrected_unit'],
                     'electron / adu')

        if inplace:
            assert_is(res, image)
        else:
            assert_is_not(res, image)


class Test_Processing_TrimImage:
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
        else:
            assert_is_not(trimmed, xi)

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
        else:
            assert_is_not(trimmed, yi)

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
        else:
            assert_is_not(trimmed, frame)

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
        else:
            assert_is_not(trimmed, frame)

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
        assert_equal(res1.flags, np.zeros((20, 20)))
        assert_equal(res1.unit, u.adu)

        if inplace:
            assert_is(res1.data, frame1.data)
        else:
            assert_is_not(res1.data, frame1.data)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_flat_with_norm_value(self, inplace):
        expect = np.ones((20, 20))*3
        expect[0:5, 0:5] = 3/0.5
        expect /= 2

        frame = FrameData(np.ones((20, 20))*3, unit=u.adu)
        master_flat = FrameData(np.ones((20, 20)))
        master_flat.data[0:5, 0:5] = 0.5

        res = flat_correct(frame, master_flat, norm_value=1/2, inplace=inplace)

        assert_is_instance(res, FrameData)
        assert_equal(res.data, expect)
        assert_equal(res.header['astropop flat_corrected'], True)
        assert_equal(res.flags, np.zeros((20, 20)))
        assert_equal(res.unit, u.adu)

        if inplace:
            assert_is(res, frame)
        else:
            assert_is_not(res, frame)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_flat_with_min_value(self, inplace):
        expect = np.ones((20, 20))*3

        frame = FrameData(np.ones((20, 20))*3, unit=u.adu)
        master_flat = FrameData(np.ones((20, 20)))
        master_flat.data[0:5, 0:5] = 0.5
        # everything in flat will be 1, the minimum value.

        res = flat_correct(frame, master_flat, min_value=1, inplace=inplace)

        assert_is_instance(res, FrameData)
        assert_equal(res.data, expect)
        assert_equal(res.header['astropop flat_corrected'], True)
        assert_equal(res.flags, np.zeros((20, 20)))
        assert_equal(res.unit, u.adu)

        if inplace:
            assert_is(res, frame)
        else:
            assert_is_not(res, frame)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_flat_named(self, inplace):
        frame = FrameData(np.ones((20, 20))*3, unit=u.adu)
        master_flat = FrameData(np.ones((20, 20)),
                                origin_filename='test_master_flat.fits')
        res = flat_correct(frame, master_flat, inplace=inplace)
        assert_equal(res.header['astropop flat_corrected'], True)
        assert_equal(res.header['astropop flat_corrected_file'],
                     'test_master_flat.fits')


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
        assert_equal(res4.flags, np.zeros((20, 20)))
        assert_equal(res4.unit, u.adu)

        if inplace:
            assert_is(res4.data, frame4bias.data)
        else:
            assert_is_not(res4.data, frame4bias.data)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_simple_bias_named(self, inplace):
        frame4bias = FrameData(np.ones((20, 20))*3, unit='adu')
        master_bias = FrameData(np.ones((20, 20)), unit='adu',
                                origin_filename='test_master_bias.fits')
        res4 = subtract_bias(frame4bias, master_bias, inplace=inplace)

        assert_is_instance(res4, FrameData)
        assert_equal(res4.header['astropop bias_corrected'], True)
        assert_equal(res4.header['astropop bias_corrected_file'],
                     'test_master_bias.fits')

        if inplace:
            assert_is(res4.data, frame4bias.data)
        else:
            assert_is_not(res4.data, frame4bias.data)


class Test_Processing_Dark():
    @pytest.mark.parametrize('inplace', [True, False])
    def test_simple_dark(self, inplace):
        expected = np.ones((20, 20))*2
        expected[0:5, 0:5] = 2.5

        frame4dark = FrameData(np.ones((20, 20))*3, unit='adu')

        master_dark = FrameData(np.ones((20, 20)), unit='adu')
        master_dark.data[0:5, 0:5] = 0.5

        res4 = subtract_dark(frame4dark, master_dark, inplace=inplace,
                             dark_exposure=1, image_exposure=1)

        assert_is_instance(res4, FrameData)
        assert_equal(res4.data, expected)
        assert_equal(res4.header['astropop dark_corrected'], True)
        assert_equal(res4.flags, np.zeros((20, 20)))
        assert_equal(res4.unit, u.adu)

        if inplace:
            assert_is(res4.data, frame4dark.data)
        else:
            assert_is_not(res4.data, frame4dark.data)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_simple_dark_scaling(self, inplace):
        expected = np.ones((20, 20))*2.5
        expected[0:5, 0:5] = 2.75

        frame4dark = FrameData(np.ones((20, 20))*3, unit='adu')
        master_dark = FrameData(np.ones((20, 20)), unit='adu')
        master_dark.data[0:5, 0:5] = 0.5

        res4 = subtract_dark(frame4dark, master_dark, inplace=inplace,
                             dark_exposure=2, image_exposure=1)

        assert_is_instance(res4, FrameData)
        assert_equal(res4.data, expected)
        assert_equal(res4.header['astropop dark_corrected'], True)
        assert_equal(res4.flags, np.zeros((20, 20)))
        assert_equal(res4.unit, u.adu)

        if inplace:
            assert_is(res4.data, frame4dark.data)
        else:
            assert_is_not(res4.data, frame4dark.data)
