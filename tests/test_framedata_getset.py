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


class Test_FrameData_GetSet_WCS:
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


class Test_FrameData_GetSet_Meta:
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
