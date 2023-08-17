# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
from astropop.image.processing import cosmics_lacosmic
from astropop.framedata import FrameData, PixelMaskFlags
from astropop.math import QFloat
from astropop.testing import *
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

