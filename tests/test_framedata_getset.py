# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropop.testing import *
from astropop.framedata import FrameData
from .test_framedata import create_framedata, DEFAULT_HEADER


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
