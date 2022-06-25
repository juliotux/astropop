# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import numpy as np
import pytest

from astropop.image.imarith import imarith
from astropop.framedata import FrameData
from astropop.testing import *


pars = pytest.mark.parametrize('op,vs', [('+', {'f1': {'v': 30, 'u': 0},
                                                'f2': {'v': 10, 'u': 0},
                                                'r': {'v': 40, 'u': 0}}),
                                         ('+', {'f1': {'v': 30, 'u': 3},
                                                'f2': {'v': 10, 'u': 4},
                                                'r': {'v': 40, 'u': 5}}),
                                         ('-', {'f1': {'v': 30, 'u': 0},
                                                'f2': {'v': 10, 'u': 0},
                                                'r': {'v': 20, 'u': 0}}),
                                         ('-', {'f1': {'v': 30, 'u': 3},
                                                'f2': {'v': 10, 'u': 4},
                                                'r': {'v': 20, 'u': 5}}),
                                         ('*', {'f1': {'v': 5, 'u': 0},
                                                'f2': {'v': 6, 'u': 0},
                                                'r': {'v': 30, 'u': 0}}),
                                         ('*', {'f1': {'v': 5, 'u': 0.3},
                                                'f2': {'v': 6, 'u': 0.4},
                                                'r': {'v': 30,
                                                      'u': 2.690725}}),
                                         ('/', {'f1': {'v': 10, 'u': 0},
                                                'f2': {'v': 3, 'u': 0},
                                                'r': {'v': 3.33333333,
                                                      'u': 0}}),
                                         ('/', {'f1': {'v': 10, 'u': 1},
                                                'f2': {'v': 3, 'u': 0.3},
                                                'r': {'v': 3.33333333,
                                                      'u': 0.47140452}}),
                                         ('//', {'f1': {'v': 10, 'u': 0},
                                                 'f2': {'v': 3, 'u': 0},
                                                 'r': {'v': 3.000000,
                                                       'u': 0}}),
                                         ('//', {'f1': {'v': 10, 'u': 1},
                                                 'f2': {'v': 3, 'u': 0.3},
                                                 'r': {'v': 3.000000,
                                                       'u': 0.000000}})])


@pytest.mark.parametrize('handle_mask', [True, False])
@pytest.mark.parametrize('inplace', [True, False])
@pars
def test_imarith_ops_frames(op, vs, inplace, handle_mask):
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

    res = imarith(frame1, frame2, op, inplace=inplace,
                  join_masks=handle_mask)

    assert_almost_equal(res.data, exp_res.data)
    assert_almost_equal(res.uncertainty, exp_res.uncertainty)
    if handle_mask:
        assert_equal(res.mask, exp_res.mask)

    if inplace:
        assert_is(res, frame1)
    else:
        assert_is_not(res, frame1)
