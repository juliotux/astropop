# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from enum import Flag
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropop.testing import *
from astropop.framedata import FrameData, PixelMaskFlags
from .test_framedata import create_framedata, DEFAULT_HEADER, DEFAULT_DATA_SIZE


class DummyFlag(Flag):
    TEST = 1


class TestFrameDataGetSetWCS:
    def test_wcs_invalid(self):
        frame = create_framedata()
        with pytest.raises(TypeError):
            frame.wcs = 5

    def test_wcs_assign(self):
        wcs = WCS(naxis=2)
        frame = create_framedata()
        frame.wcs = wcs
        assert_equal(frame.wcs, wcs)

    def test_framedata_set_wcs_none(self):
        frame = create_framedata()
        frame.wcs = None
        assert_equal(frame.wcs, None)

    def test_framedata_set_wcs(self):
        frame = create_framedata()
        wcs = WCS(naxis=2)
        frame.wcs = wcs
        assert_equal(frame.wcs, wcs)

    def test_framedata_set_wcs_error(self):
        frame = create_framedata()
        with pytest.raises(TypeError,
                           match='wcs setter value must be a WCS instance.'):
            frame.wcs = 1


class TestFrameDataGetSetMeta:
    def test_get_meta_is_header(self):
        frame = create_framedata()
        assert_is_instance(frame.meta, fits.Header)

    def test_get_header_is_header(self):
        frame = create_framedata()
        assert_is_instance(frame.header, fits.Header)

    def test_set_meta_dict(selt):
        frame = create_framedata()
        frame.meta = DEFAULT_HEADER
        for i in DEFAULT_HEADER.keys():
            assert_equal(DEFAULT_HEADER[i], frame.meta[i])

    def test_set_meta_header(self):
        frame = create_framedata()
        frame.meta = fits.Header(DEFAULT_HEADER)
        for i in DEFAULT_HEADER.keys():
            assert_equal(DEFAULT_HEADER[i], frame.meta[i])


class TestFrameDataGetSetUnit:
    def test_set_data_no_unit(self):
        frame = create_framedata()
        frame.data = 1
        assert_equal(frame.data, 1)
        assert_equal(frame.unit, 'adu')

    def test_set_data_with_unit(self):
        frame = create_framedata()
        frame.data = 1*u.Unit('s')
        assert_equal(frame.data, 1)
        assert_equal(frame.unit, 's')


class TestFrameDataGetSetMathProps:
    def test_framedata_median_without_unit(self):
        frame = FrameData([[5, 1, 3, 4, 1]])
        assert_equal(frame.median(), 3)

    def test_framedata_median_with_unit(self):
        frame = FrameData([[5, 1, 3, 4, 1]], unit='adu')
        assert_equal(frame.median(), 3*u.Unit('adu'))

    def test_framedata_mean_without_unit(self):
        frame = FrameData([[2, 1, 3, 8, 6]])
        assert_equal(frame.mean(), 4)

    def test_framedata_mean_with_unit(self):
        frame = FrameData([[2, 1, 3, 8, 6]], unit='adu')
        assert_equal(frame.mean(), 4*u.Unit('adu'))

    def test_framedata_min_without_unit(self):
        frame = FrameData([[2, 1, 3, 8, 6]])
        assert_equal(frame.min(), 1)

    def test_framedata_min_with_unit(self):
        frame = FrameData([[2, 1, 3, 8, 6]], unit='adu')
        assert_equal(frame.min(), 1*u.Unit('adu'))

    def test_framedata_max_without_unit(self):
        frame = FrameData([[1, 2, 3, 8, 6]])
        assert_equal(frame.max(), 8)

    def test_framedata_max_with_unit(self):
        frame = FrameData([[1, 2, 3, 8, 6]], unit='adu')
        assert_equal(frame.max(), 8*u.Unit('adu'))

    def test_framedata_std_without_unit(self):
        frame = FrameData([np.arange(10)])
        res = frame.std()
        assert_almost_equal(res, 2.8722813232690143)

    def test_framedata_std_with_unit(self):
        frame = FrameData([np.arange(10)], unit='adu')
        res = frame.std()
        assert_almost_equal(res.value, 2.8722813232690143)
        assert_equal(res.unit, u.adu)

    def test_framedata_statistics_without_unit(self):
        frame = FrameData([np.arange(9)])
        res = frame.statistics()
        assert_equal(res['mean'], 4)
        assert_equal(res['median'], 4)
        assert_equal(res['min'], 0)
        assert_equal(res['max'], 8)
        assert_almost_equal(res['std'], 2.581988897471611)

    def test_framedata_statistics_with_unit(self):
        frame = FrameData([np.arange(9)], unit='adu')
        res = frame.statistics()
        assert_equal(res['mean'].value, 4)
        assert_equal(res['median'].value, 4)
        assert_equal(res['min'].value, 0)
        assert_equal(res['max'].value, 8)
        assert_almost_equal(res['std'].value, 2.581988897471611)
        assert_equal(res['mean'].unit, u.adu)
        assert_equal(res['median'].unit, u.adu)
        assert_equal(res['min'].unit, u.adu)
        assert_equal(res['max'].unit, u.adu)


class TestFrameDataGetSetUncertainty:
    def test_setting_uncertainty_with_array(self):
        frame = create_framedata()
        frame.uncertainty = None
        fake_uncertainty = np.sqrt(np.abs(frame.data))
        frame.uncertainty = fake_uncertainty.copy()
        assert_equal(frame.uncertainty, fake_uncertainty)
        assert_equal(frame.unit, u.adu)

    def test_setting_uncertainty_with_scalar(self):
        uncertainty = 10
        frame = create_framedata()
        frame.uncertainty = None
        frame.uncertainty = uncertainty
        fake_uncertainty = np.zeros_like(frame.data)
        fake_uncertainty[:] = uncertainty
        assert_equal(frame.uncertainty, fake_uncertainty)
        assert_equal(frame.unit, u.adu)

    def test_setting_uncertainty_with_quantity(self):
        uncertainty = 10*u.adu
        frame = create_framedata()
        frame.uncertainty = None
        frame.uncertainty = uncertainty
        fake_uncertainty = np.zeros_like(frame.data)
        fake_uncertainty[:] = uncertainty.value
        assert_equal(frame.uncertainty, fake_uncertainty)
        assert_equal(frame.unit, u.adu)

    def test_setting_uncertainty_wrong_shape_raises_error(self):
        frame = create_framedata()
        with pytest.raises(ValueError):
            frame.uncertainty = np.zeros([3, 4])

    def test_setting_bad_uncertainty_raises_error(self):
        frame = create_framedata()
        with pytest.raises(ValueError, match='could not convert'):
            # Uncertainty is supposed to be an instance of NDUncertainty
            frame.uncertainty = 'not a uncertainty'

    def test_none_uncertainty_returns_empty(self):
        frame = create_framedata()
        assert_is_none(frame.uncertainty)

    def test_get_uncertainty_empty_return_none(self):
        # test the get_uncertainty method with return_none=True
        frame = create_framedata()
        assert_is_none(frame.get_uncertainty(True), None)

    def test_get_uncertainty_empty_return_zero(self):
        # test the get_uncertainty method with return_none=False
        frame = create_framedata()
        shp = (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)
        assert_equal(frame.get_uncertainty(False), np.zeros(shp))

    def test_get_uncertainty_non_empty_return_none(self):
        # test the get_uncertainty method with return_none=True
        shp = (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)
        frame = create_framedata(uncertainty=np.ones(shp))
        assert_equal(frame.get_uncertainty(True), np.ones(shp))

    def test_get_uncertainty_non_empty_return_zero(self):
        # test the get_uncertainty method with return_none=False
        shp = (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)
        frame = create_framedata(uncertainty=np.ones(shp))
        assert_equal(frame.get_uncertainty(False), np.ones(shp))

    def test_set_uncertainty_memmapped(self):
        # test that setting uncertainty on a memmapped FrameData
        # does not change the data
        frame = create_framedata()
        frame.enable_memmap()
        frame.uncertainty = np.ones_like(frame.data)
        assert_equal(frame.uncertainty, np.ones_like(frame.data))


class TestFrameDataGetSetFlags:
    def test_setting_flags_None(self):
        frame = create_framedata()
        frame.flags = None
        assert_is_none(frame.flags)

    def test_setting_flags_with_array(self):
        frame = create_framedata()
        fake_flags = np.ones_like(frame.data, dtype='uint8')
        frame.flags = fake_flags.copy()
        assert_equal(frame.flags, fake_flags)

    def test_setting_flags_with_scalar_error(self):
        frame = create_framedata()
        with pytest.raises(ValueError, match='Flags cannot be scalar.'):
            frame.flags = 1

    def test_setting_flags_with_quantity_error(self):
        frame = create_framedata()
        with pytest.raises(ValueError, match='Flags cannot have units.'):
            frame.flags = np.zeros_like(frame.data, dtype='uint8')*u.adu

    def test_setting_flags_wrong_shape_raises_error(self):
        frame = create_framedata()
        with pytest.raises(ValueError):
            frame.flags = np.zeros([3, 4])

    def test_setting_bad_flags_raises_error(self):
        frame = create_framedata()
        with pytest.raises(ValueError):
            frame.flags = 'not a flags'

    def test_set_uncertainty_memmapped(self):
        # test that setting uncertainty on a memmapped FrameData
        # does not change the data
        frame = create_framedata()
        frame.enable_memmap()
        frame.flags = np.ones_like(frame.data, dtype='uint8')
        assert_equal(frame.flags, np.ones_like(frame.data))

    def test_add_flags_bool_array_where(self):
        frame = create_framedata()
        where = np.zeros_like(frame.data, dtype=bool)
        where[0, 0] = True
        frame.add_flags(PixelMaskFlags.MASKED, where)
        expected = np.zeros_like(frame.data, dtype='uint8')
        expected[0, 0] = PixelMaskFlags.MASKED.value
        assert_equal(frame.flags, expected)

    def test_add_flags_index_where(self):
        frame = create_framedata()
        frame.add_flags(PixelMaskFlags.MASKED, (0, 0))
        expected = np.zeros_like(frame.data, dtype='uint8')
        expected[0, 0] = PixelMaskFlags.MASKED.value
        assert_equal(frame.flags, expected)

    def test_add_flags_index_where_2d(self):
        frame = create_framedata()
        frame.add_flags(PixelMaskFlags.MASKED, ([0, 2], [0, 1]))
        expected = np.zeros_like(frame.data, dtype='uint8')
        expected[0, 0] = PixelMaskFlags.MASKED.value
        expected[2, 1] = PixelMaskFlags.MASKED.value
        assert_equal(frame.flags, expected)

    @pytest.mark.parametrize('flag', [1, 'masked', DummyFlag.TEST])
    def test_add_flags_error_type(self, flag):
        frame = create_framedata()
        with pytest.raises(TypeError, match='PixelMaskFlags'):
            frame.add_flags(1, (0, 0))


class TestFrameDataMaskedData:
    def test_masked_simple(self):
        f = FrameData([[1, 2, 3, 4, 5]],
                      mask=[[True, False, False, False, True]])
        assert_equal(f.get_masked_data(),
                     np.array([[np.nan, 2, 3, 4, np.nan]]))

    def test_masked_fill_value(self):
        f = FrameData([[1, 2, 3, 4, 5]],
                      mask=[[True, False, False, False, True]])
        assert_equal(f.get_masked_data(fill_value=0),
                     np.array([[0, 2, 3, 4, 0]]))
