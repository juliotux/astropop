# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import numpy.testing as npt
import pytest
import pytest_check as check

from astropop.image_processing.imarith import imarith
from astropop.framedata import FrameData


# TODO: Continue the tests
# TODO: Test with None FrameData
# TODO: Test with None scalar values


def test_invalid_op():
    frame1 = FrameData(np.zeros((10, 10)), unit='')
    frame2 = FrameData(np.zeros((10, 10)), unit='')
    with pytest.raises(ValueError) as exc:
        imarith(frame1, frame2, 'not an op')
        check.is_in('not supported', str(exc.value))

# session for test with operations ('+', '-') that use function <np.sqrt(sa/a**2) + (sb/b**2)>.
# operation '+'

def test_invalid_shapes():
    frame1 = FrameData(np.zeros((10, 10)), unit='')
    frame2 = FrameData(np.zeros((5, 5)), unit='')
    with pytest.raises(ValueError):
        imarith(frame1, frame2, '+')


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_add(inplace):
    frame1 = FrameData(np.ones((10, 10)), unit='adu')
    frame2 = FrameData(np.ones((10, 10)), unit='adu')
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 40
    res = imarith(frame1, frame2, '+', propagate_errors=False,
                  handle_mask=False, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_add_uncertainty(inplace):
    frame1 = FrameData(np.ones((10, 10)), unit='adu',
                       uncertainty=3.0, u_unit='adu')
    frame2 = FrameData(np.ones((10, 10)), unit='adu',
                       uncertainty=4.0, u_unit='adu')
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 40
    res = imarith(frame1, frame2, '+', propagate_errors=True,
                  handle_mask=False, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    npt.assert_array_almost_equal(res.uncertainty,
                                  np.ones_like(frame2.data)*5.0)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_add_mask(inplace):
    mask1 = np.zeros((10, 10))
    mask2 = np.zeros((10, 10))
    mask1[5, 5] = 1
    mask2[3, 3] = 1
    expect = np.zeros((10, 10))
    expect[5, 5] = 1
    expect[3, 3] = 1
    frame1 = FrameData(np.ones((10, 10)), unit='adu',
                       mask=mask1)
    frame2 = FrameData(np.ones((10, 10)), unit='adu',
                       mask=mask2)
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 40
    res = imarith(frame1, frame2, '+', propagate_errors=False,
                  handle_mask=True, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    npt.assert_array_almost_equal(res.mask, expect)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)

 # operation '-'

def test_invalid_shapes():
    frame1 = FrameData(np.zeros((10, 10)), unit='')
    frame2 = FrameData(np.zeros((5, 5)), unit='')
    with pytest.raises(ValueError):
        imarith(frame1, frame2, '-')


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_sub(inplace):
    frame1 = FrameData(np.ones((10, 10)), unit='adu')
    frame2 = FrameData(np.ones((10, 10)), unit='adu')
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 40
    res = imarith(frame1, frame2, '-', propagate_errors=False,
                  handle_mask=False, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_sub_uncertainty(inplace):
    frame1 = FrameData(np.ones((10, 10)), unit='adu',
                       uncertainty=3.0, u_unit='adu')
    frame2 = FrameData(np.ones((10, 10)), unit='adu',
                       uncertainty=4.0, u_unit='adu')
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 20
    res = imarith(frame1, frame2, '-', propagate_errors=True,
                  handle_mask=False, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    npt.assert_array_almost_equal(res.uncertainty,
                                  np.ones_like(frame2.data)*5.0)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_sub_mask(inplace):
    mask1 = np.zeros((10, 10))
    mask2 = np.zeros((10, 10))
    mask1[5, 5] = 1
    mask2[3, 3] = 1
    expect = np.zeros((10, 10))
    expect[5, 5] = 1
    expect[3, 3] = 1
    frame1 = FrameData(np.ones((10, 10)), unit='adu',
                       mask=mask1)
    frame2 = FrameData(np.ones((10, 10)), unit='adu',
                       mask=mask2)
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 20
    res = imarith(frame1, frame2, '-', propagate_errors=False,
                  handle_mask=True, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    npt.assert_array_almost_equal(res.mask, expect)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)

# session for test with operations ('*', '/', '//') that use function <f*np.sqrt((sa/a)**2 + (sb/b)**2))>.
# operation '*'


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_mlp(inplace):
    frame1 = FrameData(np.ones((10, 10)), unit='adu')
    frame2 = FrameData(np.ones((10, 10)), unit='adu')
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 300
    res = imarith(frame1, frame2, '*', propagate_errors=False,
                  handle_mask=False, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_mlp_uncertainty(inplace):
    frame1 = FrameData(np.ones((10, 10)), unit='adu',
                       uncertainty=3.0, u_unit='adu')
    frame2 = FrameData(np.ones((10, 10)), unit='adu',
                       uncertainty=4.0, u_unit='adu')
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 300
    res = imarith(frame1, frame2, '*', propagate_errors=True,
                  handle_mask=False, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    npt.assert_array_almost_equal(res.uncertainty,
                                  np.ones_like(frame2.data)*5.0)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_mlp_mask(inplace):
    mask1 = np.zeros((10, 10))
    mask2 = np.zeros((10, 10))
    mask1[5, 5] = 1
    mask2[3, 3] = 1
    expect = np.zeros((10, 10))
    expect[5, 5] = 1
    expect[3, 3] = 1
    frame1 = FrameData(np.ones((10, 10)), unit='adu',
                       mask=mask1)
    frame2 = FrameData(np.ones((10, 10)), unit='adu',
                       mask=mask2)
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 300
    res = imarith(frame1, frame2, '*', propagate_errors=False,
                  handle_mask=True, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    npt.assert_array_almost_equal(res.mask, expect)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)

# operation '/'

@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_div(inplace):
    frame1 = FrameData(np.ones((10, 10)), unit='adu')
    frame2 = FrameData(np.ones((10, 10)), unit='adu')
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 3
    res = imarith(frame1, frame2, '/', propagate_errors=False,
                  handle_mask=False, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_div_uncertainty(inplace):
    frame1 = FrameData(np.ones((10, 10)), unit='adu',
                       uncertainty=3.0, u_unit='adu')
    frame2 = FrameData(np.ones((10, 10)), unit='adu',
                       uncertainty=4.0, u_unit='adu')
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 3
    res = imarith(frame1, frame2, '/', propagate_errors=True,
                  handle_mask=False, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    npt.assert_array_almost_equal(res.uncertainty,
                                  np.ones_like(frame2.data)*5.0)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_div_mask(inplace):
    mask1 = np.zeros((10, 10))
    mask2 = np.zeros((10, 10))
    mask1[5, 5] = 1
    mask2[3, 3] = 1
    expect = np.zeros((10, 10))
    expect[5, 5] = 1
    expect[3, 3] = 1
    frame1 = FrameData(np.ones((10, 10)), unit='adu',
                       mask=mask1)
    frame2 = FrameData(np.ones((10, 10)), unit='adu',
                       mask=mask2)
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 3
    res = imarith(frame1, frame2, '/', propagate_errors=False,
                  handle_mask=True, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    npt.assert_array_almost_equal(res.mask, expect)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)

# operation '//'

@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_dva(inplace):
    frame1 = FrameData(np.ones((10, 10)), unit='adu')
    frame2 = FrameData(np.ones((10, 10)), unit='adu')
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 3
    res = imarith(frame1, frame2, '//', propagate_errors=False,
                  handle_mask=False, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_dva_uncertainty(inplace):
    frame1 = FrameData(np.ones((10, 10)), unit='adu',
                       uncertainty=3.0, u_unit='adu')
    frame2 = FrameData(np.ones((10, 10)), unit='adu',
                       uncertainty=4.0, u_unit='adu')
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 3
    res = imarith(frame1, frame2, '//', propagate_errors=True,
                  handle_mask=False, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    npt.assert_array_almost_equal(res.uncertainty,
                                  np.ones_like(frame2.data)*5.0)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)


@pytest.mark.parametrize('inplace', [True, False])
def test_imarith_dva_mask(inplace):
    mask1 = np.zeros((10, 10))
    mask2 = np.zeros((10, 10))
    mask1[5, 5] = 1
    mask2[3, 3] = 1
    expect = np.zeros((10, 10))
    expect[5, 5] = 1
    expect[3, 3] = 1
    frame1 = FrameData(np.ones((10, 10)), unit='adu',
                       mask=mask1)
    frame2 = FrameData(np.ones((10, 10)), unit='adu',
                       mask=mask2)
    frame2.data[:] = 10
    frame1.data[:] = 30
    exp_res = np.ones((10, 10))
    exp_res[:] = 3
    res = imarith(frame1, frame2, '//', propagate_errors=False,
                  handle_mask=True, inplace=inplace)
    npt.assert_array_equal(res.data, exp_res)
    npt.assert_array_almost_equal(res.mask, expect)
    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)

# session for test with operations ('**', '%') that use function <f*b*sa/a>.
# operation '**'

        