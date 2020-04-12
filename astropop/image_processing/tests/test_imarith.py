# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import numpy.testing as npt
import pytest
import pytest_check as check

from astropop.image_processing.imarith import imarith
from astropop.framedata import FrameData


# TODO: Test with None FrameData
# TODO: Test with None scalar values


# TODO: '%' and '**' functions
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('handle_mask', [True, False])
@pytest.mark.parametrize('op,values', [('+', {'f1': {'v': 30, 'u': None},
                                              'f2': {'v': 10, 'u': None},
                                              'r': {'v': 40, 'u': None}}),
                                       ('+', {'f1': {'v': 30, 'u': 3},
                                              'f2': {'v': 10, 'u': 4},
                                              'r': {'v': 40, 'u': 5}}),
                                       ('-', {'f1': {'v': 30, 'u': None},
                                              'f2': {'v': 10, 'u': None},
                                              'r': {'v': 20, 'u': None}}),
                                       ('-', {'f1': {'v': 30, 'u': 3},
                                              'f2': {'v': 10, 'u': 4},
                                              'r': {'v': 20, 'u': 5}}),
                                       ('*', {'f1': {'v': 5, 'u': None},
                                              'f2': {'v': 6, 'u': None},
                                              'r': {'v': 30, 'u': None}}),
                                       ('*', {'f1': {'v': 5, 'u': 0.3},
                                              'f2': {'v': 6, 'u': 0.4},
                                              'r': {'v': 30, 'u': 2.0}}),
                                       ('/', {'f1': {'v': 10, 'u': None},
                                              'f2': {'v': 3, 'u': None},
                                              'r': {'v': 3.33333333,
                                                    'u': None}}),
                                       ('/', {'f1': {'v': 10, 'u': 1},
                                              'f2': {'v': 3, 'u': 0.3},
                                              'r': {'v': 3.33333333,
                                                    'u': 0.47140452}}),
                                       ('//', {'f1': {'v': 10, 'u': None},
                                               'f2': {'v': 3, 'u': None},
                                               'r': {'v': 3.000000,
                                                     'u': None}}),
                                       ('//', {'f1': {'v': 10, 'u': 1},
                                               'f2': {'v': 3, 'u': 0.3},
                                               'r': {'v': 3.000000,
                                                     'u': 0.424264}})])
def test_imarith_ops_frames(op, values, inplace, handle_mask):
    propagate_errors = [False]  # use list to gen_frame works

    def gen_frame(v):
        # Gen frames with {'v', 'u'} dict
        shape = (10, 10)
        if v['u'] is None:
            frame = FrameData(np.ones(shape, dtype='f8'), unit='adu')
        else:
            frame = FrameData(np.ones(shape, dtype='f8'), unit='adu',
                              uncertainty=v['u'],
                              u_unit='adu')
            propagate_errors[0] = True
        frame.data[:] = v['v']
        return frame

    frame1 = gen_frame(values['f1'])
    frame2 = gen_frame(values['f2'])
    exp_res = gen_frame(values['r'])
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

    propagate_errors = propagate_errors[0]
    res = imarith(frame1, frame2, op, inplace=inplace,
                  propagate_errors=propagate_errors,
                  handle_mask=handle_mask)

    npt.assert_array_almost_equal(res.data, exp_res.data)
    if propagate_errors:
        npt.assert_array_almost_equal(res.uncertainty,
                                      exp_res.uncertainty)
    if handle_mask:
        npt.assert_array_equal(res.mask, exp_res.mask)

    if inplace:
        check.is_true(res is frame1)
    else:
        check.is_false(res is frame1)


def test_invalid_op():
    frame1 = FrameData(np.zeros((10, 10)), unit='')
    frame2 = FrameData(np.zeros((10, 10)), unit='')
    with pytest.raises(ValueError) as exc:
        imarith(frame1, frame2, 'not an op')
        check.is_in('not supported', str(exc.value))


def test_invalid_shapes():
    frame1 = FrameData(np.zeros((10, 10)), unit='')
    frame2 = FrameData(np.zeros((5, 5)), unit='')
    with pytest.raises(ValueError):
        imarith(frame1, frame2, '+')
