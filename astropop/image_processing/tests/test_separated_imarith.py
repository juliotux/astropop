# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from astropop.image_processing.imarith import imarith
from astropop.framedata import FrameData
from astropop.testing import assert_equal, assert_is, assert_is_not, \
                             assert_in

# TODO: Test with None FrameData
# TODO: Test with None scalar values

# TODO: '%' and '**' functions
# pars = pytest.mark.parametrize('vs', [({'f1': {'v': 30, 'u': None},
#                                                 'f2': {'v': 00, 'u': None},
#                                                 'r': {'v': 30, 'u': None}}),
#                                           ({'f1': {'v': 30, 'u': 3},
#                                                  'f2': {'v': 10, 'u': 4},
#                                                  'r': {'v': 40, 'u': 5}}),
#                                           ({'f1': {'v': 30, 'u': 3},
#                                                  'f2': {'v': 00, 'u': 4},
#                                                  'r': {'v': 30, 'u': 5}}),
#                                          ({'f1': {'v': 00, 'u': None},
#                                                 'f2': {'v': 10, 'u': None},
#                                                 'r': {'v': 10, 'u': None}})])                                    

@pytest.mark.parametrize('handle_mask', [True, False])
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('vs', [({'f1': {'v': 30, 'u': 0},
                                  'f2': {'v': 0, 'u': 0},
                                  'r': {'v': 30, 'u': 0}}),
                                ({'f1': {'v': 0, 'u': 3},
                                  'f2': {'v': 10, 'u': 4},
                                  'r': {'v': 10, 'u': 5}})])
def test_separated_sum_imarith_ops_frames(vs, inplace, handle_mask):
    frame1 = gen_frame(vs['f1'])
    frame2 = gen_frame(vs['f2'])
    exp_res = gen_frame(vs['r'])
    if handle_mask:
        mask1 = np.zeros((10, 10))
        mask2 = np.zeros((10, 10))
        mask1[5, 5] = 1
        mask2[3, 3] = 1
        exp_mask = np.zeros((10, 10))
        exp_mask[5, 5] = 1
        exp_mask[3, 3] = 1
        frame1.mask = mask1
        frame2.mask = mask2
        exp_res.mask = exp_mask

    op = '+'    
    res = imarith(frame1, frame2, op, inplace=inplace,
                  join_masks=handle_mask)

    assert_equal(res.data, exp_res.data)
    assert_equal(res.uncertainty, exp_res.uncertainty)
    if handle_mask:
        assert_equal(res.mask, exp_res.mask)

    if inplace:
        assert_is(res,frame1)
    else:
        assert_is_not(res,frame1)


def test_invalid_op():
    frame1 = FrameData(np.zeros((10, 10)), unit='')
    frame2 = FrameData(np.zeros((10, 10)), unit='')
    with pytest.raises(ValueError) as exc:
        imarith(frame1, frame2, 'not an op')
        assert_in('not supported', str(exc.value))


def test_invalid_shapes():
    frame1 = FrameData(np.zeros((10, 10)), unit='')
    frame2 = FrameData(np.zeros((5, 5)), unit='')
    with pytest.raises(ValueError):
        imarith(frame1, frame2, '+')

def gen_frame(v):
    # Gen frames with {'v', 'u'} dict
    shape = (10, 10)
    if v['u'] is None:
        frame = FrameData(np.ones(shape, dtype='f8'), unit='adu')
    else:
        frame = FrameData(np.ones(shape, dtype='f8'), unit='adu',
                          uncertainty=v['u'])
    frame.data[:] = v['v']
    return frame
