# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import numpy.testing as npt
import pytest

from astropop.image_processing.imarith import imarith
from astropop.framedata import FrameData


# TODO: Continue the tests


def test_invalid_op():
    frame1 = FrameData(np.zeros((10, 10)), unit='')
    frame2 = FrameData(np.zeros((10, 10)), unit='')
    with pytest.raises(ValueError) as exc:
        imarith(frame1, frame2, 'not an op')
        assert 'not suppoerted' in str(exc.value)


def test_invalid_shapes():
    frame1 = FrameData(np.zeros((10, 10)), unit='')
    frame2 = FrameData(np.zeros((5, 5)), unit='')
    with pytest.raises(ValueError):
        imarith(frame1, frame2, '+')


def test_imarith_add():
    frame1 = FrameData(np.ones((10, 10)), unit='adu')
    frame2 = FrameData(np.ones((10, 10)), unit='adu')
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 40
    res = imarith(frame1, frame2, '+')
    npt.assert_array_equal(res.data, exp_res)
