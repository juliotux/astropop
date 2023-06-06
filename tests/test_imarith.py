# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import numpy as np
import pytest

from astropop.image.imarith import imarith
from astropop.framedata import FrameData
from astropop.math.physical import UnitsError, units, QFloat
from astropop.testing import *


# TODO: % and ** functions


SHAPE = (10, 10)


def gen_frame(value, uncertainty=None, unit='adu'):
    """Gen frames with {'v', 'u'} dict"""
    shape = SHAPE
    frame = FrameData(np.ones(shape, dtype='f8'), unit=unit,
                      uncertainty=uncertainty)
    frame.data[:] = value
    return frame


def std_test_frame(op, frame1, frame2, result, inplace, handle_flags):
    """Standart comparison tests with frames."""
    if handle_flags:
        # Use two different masks to check if they are compined
        flag1 = np.zeros(SHAPE, dtype=np.uint8)
        flag1[2, 2] = 2
        flag1[1, 1] = 1
        frame1.flags = flag1

        if isinstance(frame2, FrameData):
            # If frame2 is qfloat, quantity or a number, it don't have mask
            flag2 = np.zeros(SHAPE, dtype=np.uint8)
            flag2[3, 3] = 1
            flag2[2, 2] = 6
            frame2.flags = flag2

        exp_flag = np.zeros(SHAPE, dtype=np.uint8)
        if handle_flags == 'or':
            exp_flag[1, 1] = 1
            exp_flag[2, 2] = 6
            exp_flag[3, 3] = 1
        elif handle_flags == 'and':
            exp_flag[2, 2] = 2

        result.flags = exp_flag

    res = imarith(frame1, frame2, op, inplace=inplace,
                  merge_flags=handle_flags)

    assert_equal(res.data, result.data)
    assert_almost_equal(res.get_uncertainty(False),
                        result.get_uncertainty(False))
    assert_equal(res.unit, result.unit)
    if handle_flags:
        assert_equal(res.flags, result.flags)

    if inplace:
        assert_is(res, frame1)
    else:
        assert_is_not(res, frame1)


@pytest.mark.parametrize('handle_flags', ['and', 'or', 'no_merge'])
@pytest.mark.parametrize('inplace', [True, False])
class Test_Imartih_OPs_add:
    op = '+'

    def test_uncertainty_all_none(self, inplace, handle_flags):
        frame1 = gen_frame(30, None, unit='adu')
        frame2 = gen_frame(10, None, unit='adu')
        result = gen_frame(40, None, unit='adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f1_none(self, inplace, handle_flags):
        frame1 = gen_frame(30, None, unit='adu')
        frame2 = gen_frame(10, 0, unit='adu')
        result = gen_frame(40, 0, unit='adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f2_none(self, inplace, handle_flags):
        frame1 = gen_frame(30, 0, unit='adu')
        frame2 = gen_frame(10, None, unit='adu')
        result = gen_frame(40, 0, unit='adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f1_set(self, inplace, handle_flags):
        frame1 = gen_frame(30, 3, unit='adu')
        frame2 = gen_frame(10, None, unit='adu')
        result = gen_frame(40, 3, unit='adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f2_set(self, inplace, handle_flags):
        frame1 = gen_frame(30, 0, unit='adu')
        frame2 = gen_frame(10, 3, unit='adu')
        result = gen_frame(40, 3, unit='adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_all_set(self, inplace, handle_flags):
        frame1 = gen_frame(30, 3, unit='adu')
        frame2 = gen_frame(10, 4, unit='adu')
        result = gen_frame(40, 5, unit='adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_zero(self, inplace, handle_flags):
        frame1 = gen_frame(10, unit='adu')
        result = gen_frame(10, unit='adu')

        # pure number should fail
        with pytest.raises(UnitsError):
            std_test_frame(self.op, frame1, 0, result,
                           inplace=inplace, handle_flags=handle_flags)

        # quantity must pass
        std_test_frame(self.op, frame1, 0*units.adu, result,
                       inplace=inplace, handle_flags=handle_flags)

        # qfloat must pass
        std_test_frame(self.op, frame1, QFloat(0, unit='adu'), result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_one(self, inplace, handle_flags):
        frame1 = gen_frame(10, unit='adu')
        result = gen_frame(11, unit='adu')

        # pure number should fail
        with pytest.raises(UnitsError):
            std_test_frame(self.op, frame1, 1, result,
                           inplace=inplace, handle_flags=handle_flags)

        # quantity must pass
        std_test_frame(self.op, frame1, 1*units.adu, result,
                       inplace=inplace, handle_flags=handle_flags)

        frame1 = gen_frame(10, unit='adu')  # if not do this, result gets wrong on inplace
        # qfloat must pass
        std_test_frame(self.op, frame1, QFloat(1, unit='adu'), result,
                       inplace=inplace, handle_flags=handle_flags)


@pytest.mark.parametrize('handle_flags', ['and', 'or', 'no_merge'])
@pytest.mark.parametrize('inplace', [True, False])
class Test_Imartih_OPs_sub:
    op = '-'

    def test_uncertainty_all_none(self, inplace, handle_flags):
        frame1 = gen_frame(30, None, unit='adu')
        frame2 = gen_frame(10, None, unit='adu')
        result = gen_frame(20, None, unit='adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f1_none(self, inplace, handle_flags):
        frame1 = gen_frame(30, None, unit='adu')
        frame2 = gen_frame(10, 0, unit='adu')
        result = gen_frame(20, 0, unit='adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f2_none(self, inplace, handle_flags):
        frame1 = gen_frame(30, 0, unit='adu')
        frame2 = gen_frame(10, None, unit='adu')
        result = gen_frame(20, 0, unit='adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f1_set(self, inplace, handle_flags):
        frame1 = gen_frame(30, 3, unit='adu')
        frame2 = gen_frame(10, None, unit='adu')
        result = gen_frame(20, 3, unit='adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f2_set(self, inplace, handle_flags):
        frame1 = gen_frame(30, 0, unit='adu')
        frame2 = gen_frame(10, 3, unit='adu')
        result = gen_frame(20, 3, unit='adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_all_set(self, inplace, handle_flags):
        frame1 = gen_frame(30, 3, unit='adu')
        frame2 = gen_frame(10, 4, unit='adu')
        result = gen_frame(20, 5, unit='adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_zero(self, inplace, handle_flags):
        frame1 = gen_frame(10, unit='adu')
        result = gen_frame(10, unit='adu')

        # pure number should fail
        with pytest.raises(UnitsError):
            std_test_frame(self.op, frame1, 0, result,
                           inplace=inplace, handle_flags=handle_flags)

        # quantity must pass
        std_test_frame(self.op, frame1, 0*units.adu, result,
                       inplace=inplace, handle_flags=handle_flags)

        # qfloat must pass
        std_test_frame(self.op, frame1, QFloat(0, unit='adu'), result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_one(self, inplace, handle_flags):
        frame1 = gen_frame(10, unit='adu')
        result = gen_frame(9, unit='adu')

        # pure number should fail
        with pytest.raises(UnitsError):
            std_test_frame(self.op, frame1, 1, result,
                           inplace=inplace, handle_flags=handle_flags)

        # quantity must pass
        std_test_frame(self.op, frame1, 1*units.adu, result,
                       inplace=inplace, handle_flags=handle_flags)

        frame1 = gen_frame(10, unit='adu')  # if not do this, result gets wrong on inplace
        # qfloat must pass
        std_test_frame(self.op, frame1, QFloat(1, unit='adu'), result,
                       inplace=inplace, handle_flags=handle_flags)


@pytest.mark.parametrize('handle_flags', ['and', 'or', 'no_merge'])
@pytest.mark.parametrize('inplace', [True, False])
class Test_Imartih_OPs_mul:
    op = '*'

    def test_uncertainty_all_none(self, inplace, handle_flags):
        frame1 = gen_frame(5, None, unit='adu')
        frame2 = gen_frame(6, None, unit='electron/adu')
        result = gen_frame(30, None, unit='electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f1_none(self, inplace, handle_flags):
        frame1 = gen_frame(5, None, unit='adu')
        frame2 = gen_frame(6, 0, unit='electron/adu')
        result = gen_frame(30, 0, unit='electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f2_none(self, inplace, handle_flags):
        frame1 = gen_frame(5, 0, unit='adu')
        frame2 = gen_frame(6, None, unit='electron/adu')
        result = gen_frame(30, 0, unit='electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f1_set(self, inplace, handle_flags):
        frame1 = gen_frame(5, 0.3, unit='adu')
        frame2 = gen_frame(6, None, unit='electron/adu')
        result = gen_frame(30, 1.8, unit='electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f2_set(self, inplace, handle_flags):
        frame1 = gen_frame(5, 0, unit='adu')
        frame2 = gen_frame(6, 0.3, unit='electron/adu')
        result = gen_frame(30, 1.5, unit='electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_all_set(self, inplace, handle_flags):
        frame1 = gen_frame(5, 0.5, unit='adu')
        frame2 = gen_frame(6, 0.3, unit='electron/adu')
        result = gen_frame(30, 3.35410196624, unit='electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_zero(self, inplace, handle_flags):
        frame1 = gen_frame(10, unit='adu')
        result = gen_frame(0, unit='adu')

        # pure number must pass
        std_test_frame(self.op, frame1, 0, result,
                       inplace=inplace, handle_flags=handle_flags)

        # quantity must pass
        std_test_frame(self.op, frame1, 0*units.dimensionless_unscaled, result,
                       inplace=inplace, handle_flags=handle_flags)

        # qfloat must pass
        std_test_frame(self.op, frame1, QFloat(0), result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_one(self, inplace, handle_flags):
        frame1 = gen_frame(10, unit='adu')
        result = gen_frame(10, unit='adu')

        # pure number must pass
        std_test_frame(self.op, frame1, 1, result,
                       inplace=inplace, handle_flags=handle_flags)

        # quantity must pass
        std_test_frame(self.op, frame1, 1*units.dimensionless_unscaled,
                       result,
                       inplace=inplace, handle_flags=handle_flags)

        frame1 = gen_frame(10)  # if not do this, result gets wrong on inplace
        # qfloat must pass
        std_test_frame(self.op, frame1, QFloat(1), result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_unit_change(self, inplace, handle_flags):
        # FrameData must pass
        frame1 = gen_frame(10, unit='adu')
        frame2 = gen_frame(2, unit='electron/adu')
        result = gen_frame(20, unit='electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

        # Quantity must pass
        frame1 = gen_frame(10, unit='adu')
        frame2 = 2*units.electron/units.adu
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

        # QFloat must pass
        frame1 = gen_frame(10, unit='adu')
        frame2 = QFloat(2, unit='electron/adu')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)


@pytest.mark.parametrize('handle_flags', ['and', 'or', 'no_merge'])
@pytest.mark.parametrize('inplace', [True, False])
class Test_Imartih_OPs_div:
    op = '/'

    def test_uncertainty_all_none(self, inplace, handle_flags):
        frame1 = gen_frame(10, None, 'adu')
        frame2 = gen_frame(2, None, 'electron')
        result = gen_frame(5, None, unit='adu/electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f1_none(self, inplace, handle_flags):
        frame1 = gen_frame(10, None, unit='adu')
        frame2 = gen_frame(2, 0, unit='electron')
        result = gen_frame(5, 0, unit='adu/electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f2_none(self, inplace, handle_flags):
        frame1 = gen_frame(10, 0, unit='adu')
        frame2 = gen_frame(2, None, unit='electron')
        result = gen_frame(5, 0, unit='adu/electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f1_set(self, inplace, handle_flags):
        frame1 = gen_frame(10, 0.2, unit='adu')
        frame2 = gen_frame(2, None, unit='electron')
        result = gen_frame(5, 0.1, unit='adu/electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f2_set(self, inplace, handle_flags):
        frame1 = gen_frame(10, None, unit='adu')
        frame2 = gen_frame(2, 0.2, unit='electron')
        result = gen_frame(5, 0.5, unit='adu/electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_all_set(self, inplace, handle_flags):
        frame1 = gen_frame(10, 0.1, unit='adu')
        frame2 = gen_frame(2, 0.2, unit='electron')
        result = gen_frame(5, 0.502493781056, unit='adu/electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_one(self, inplace, handle_flags):
        frame1 = gen_frame(10, unit='adu')
        result = gen_frame(10, unit='adu')

        # pure number must pass
        std_test_frame(self.op, frame1, 1, result,
                       inplace=inplace, handle_flags=handle_flags)

        # quantity must pass
        std_test_frame(self.op, frame1, 1*units.dimensionless_unscaled,
                       result,
                       inplace=inplace, handle_flags=handle_flags)

        frame1 = gen_frame(10)  # if not do this, result gets wrong on inplace
        # qfloat must pass
        std_test_frame(self.op, frame1, QFloat(1), result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_unit_change(self, inplace, handle_flags):
        frame1 = gen_frame(10, unit='adu')
        result = gen_frame(5, unit='electron')

        # FrameData must pass
        frame2 = gen_frame(2, unit='adu/electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

        # Quantity must pass
        frame1 = gen_frame(10)
        frame2 = 2*units.adu/units.electron
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

        # QFloat must pass
        frame1 = gen_frame(10)
        frame2 = QFloat(2, unit='adu/electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)


@pytest.mark.parametrize('handle_flags', ['and', 'or', 'no_merge'])
@pytest.mark.parametrize('inplace', [True, False])
class Test_Imartih_OPs_fllordiv:
    op = '//'

    def test_uncertainty_all_none(self, inplace, handle_flags):
        frame1 = gen_frame(11, None, unit='adu')
        frame2 = gen_frame(2, None, unit='s')
        result = gen_frame(5, None, unit='adu/s')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f1_none(self, inplace, handle_flags):
        frame1 = gen_frame(11, None, unit='adu')
        frame2 = gen_frame(2, 0, unit='s')
        result = gen_frame(5, 0, unit='adu/s')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f2_none(self, inplace, handle_flags):
        frame1 = gen_frame(11, 0, unit='adu')
        frame2 = gen_frame(2, None, unit='s')
        result = gen_frame(5, 0, unit='adu/s')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f1_set(self, inplace, handle_flags):
        frame1 = gen_frame(11, 0.2, unit='adu')
        frame2 = gen_frame(2, None, unit='s')
        result = gen_frame(5, 0.0, unit='adu/s')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_f2_set(self, inplace, handle_flags):
        frame1 = gen_frame(11, None, unit='adu')
        frame2 = gen_frame(2, 0.2, unit='s')
        result = gen_frame(5, 0.0, unit='adu/s')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_uncertainty_all_set(self, inplace, handle_flags):
        frame1 = gen_frame(11, 0.1, unit='adu')
        frame2 = gen_frame(2, 0.2, unit='s')
        result = gen_frame(5, 0, unit='adu/s')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_one(self, inplace, handle_flags):
        frame1 = gen_frame(10.5, unit='adu')
        result = gen_frame(10, unit='adu')

        # pure number must pass
        std_test_frame(self.op, frame1, 1, result,
                       inplace=inplace, handle_flags=handle_flags)

        # quantity must pass
        std_test_frame(self.op, frame1, 1*units.dimensionless_unscaled,
                       result,
                       inplace=inplace, handle_flags=handle_flags)

        frame1 = gen_frame(10)  # if not do this, result gets wrong on inplace
        # qfloat must pass
        std_test_frame(self.op, frame1, QFloat(1), result,
                       inplace=inplace, handle_flags=handle_flags)

    def test_unit_change(self, inplace, handle_flags):
        frame1 = gen_frame(11, unit='adu')
        result = gen_frame(5, unit='electron')

        # FrameData must pass
        frame2 = gen_frame(2, unit='adu/electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

        # Quantity must pass
        frame1 = gen_frame(10)
        frame2 = 2*units.adu/units.electron
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)

        # QFloat must pass
        frame1 = gen_frame(10, unit='adu')
        frame2 = QFloat(2, unit='adu/electron')
        std_test_frame(self.op, frame1, frame2, result,
                       inplace=inplace, handle_flags=handle_flags)


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
