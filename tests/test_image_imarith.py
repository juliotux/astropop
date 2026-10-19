# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405
# TODO: % and ** functions


import pytest
import numpy as np

from astropop.image.imarith import ImArith, imarith
from astropop.framedata import FrameData
from astropop.math.physical import UnitsError, units, QFloat
from astropop.config import AstropopConfig as conf
from astropop.testing import *
from astropy.nddata import (CCDData, StdDevUncertainty, VarianceUncertainty,
                           InverseVariance, UnknownUncertainty)


SHAPE = (10, 10)


class Test_ImArith:
    class FrameSubclass(FrameData):
        pass

    def run(self, entry, operand1, operand2, operation, inplace=False, **options):
        if entry == 'function':
            return imarith(operand1, operand2, operation, inplace=inplace, **options)
        return ImArith(**options).run(operand1, operand2, operation, inplace=inplace)

    def operand(self, kind):
        data = np.full((2, 3), 200.)
        if kind == 'quantity':
            return data * units.cm
        if kind == 'qfloat':
            return QFloat(data, np.full(data.shape, 400.), 'cm')
        if kind.startswith('ccd'):
            errors = {'ccd_std': StdDevUncertainty(np.full(data.shape, 400.)),
                      'ccd_var': VarianceUncertainty(np.full(data.shape, 400.**2)),
                      'ccd_inv': InverseVariance(np.full(data.shape, 400.**-2)),
                      'ccd_none': None}
            return CCDData(data, unit='cm', uncertainty=errors[kind])
        cls = self.FrameSubclass if kind == 'subclass' else FrameData
        return cls(data, uncertainty=400., unit='cm')

    @pytest.mark.parametrize('entry', ['function', 'class'])
    @pytest.mark.parametrize('option', [None, 'FRAMEDATA_DISABLE_UNCERTAINTY'])
    @pytest.mark.parametrize('kind', ['frame', 'subclass', 'quantity', 'qfloat',
                                     'ccd_std', 'ccd_var', 'ccd_inv', 'ccd_none'])
    @pytest.mark.parametrize('reverse', [False, True])
    def test_operand_conversion(self, monkeypatch, entry, option, kind, reverse):
        first = FrameData(np.full((2, 3), 4.), uncertainty=3., unit='m')
        second = self.operand(kind)
        if option:
            monkeypatch.setattr(conf, option, True)
        operands = (second, first) if reverse else (first, second)
        result = self.run(entry, *operands, '+')
        scale = 100 if reverse else 1
        np.testing.assert_allclose(result.data, 6 * scale)
        assert result.unit == (units.cm if reverse else units.m)
        if option:
            assert result.uncertainty is None
        else:
            error = 3 if kind in ('quantity', 'ccd_none') else 5
            np.testing.assert_allclose(result.uncertainty, error * scale)
        np.testing.assert_array_equal(first.data, 4)
        data = second.nominal if isinstance(second, QFloat) else second.data
        if kind == 'quantity':
            data = second.value
        np.testing.assert_array_equal(data, 200)

    @pytest.mark.parametrize('entry', ['function', 'class'])
    @pytest.mark.parametrize('inplace', [False, True])
    @pytest.mark.parametrize('flag_strategy', ['or', 'and', 'no_merge'])
    @pytest.mark.parametrize('header_strategy', ['first', 'only_equal',
                                               'selected_keys', 'no_merge'])
    def test_merge_settings(self, entry, inplace, flag_strategy, header_strategy):
        header1 = {'OBJECT': 'test', 'FILTER': 'R', 'FIRST': 1}
        header2 = {'OBJECT': 'test', 'FILTER': 'V', 'SECOND': 2}
        first = FrameData(np.full((2, 3), 4.), uncertainty=3., unit='m',
                          flags=np.ones((2, 3), dtype='u1'), meta=header1)
        second = FrameData(np.full((2, 3), 2.), uncertainty=4., unit='m',
                           flags=np.full((2, 3), 3, dtype='u1'), meta=header2)
        result = self.run(entry, first, second, '+', inplace=inplace,
                          merge_flags=flag_strategy, merge_headers=header_strategy,
                          selected_keys=['FILTER', 'SECOND'])
        assert (result is first) is inplace
        np.testing.assert_array_equal(result.data, 6)
        np.testing.assert_array_equal(result.uncertainty, 5)
        np.testing.assert_array_equal(result.flags,
                                     {'or': 3, 'and': 1, 'no_merge': 0}[flag_strategy])
        expected = {'first': header1, 'only_equal': {'OBJECT': 'test'},
                    'selected_keys': {'FILTER': 'R', 'SECOND': 2}, 'no_merge': {}}
        assert dict(result.header) == expected[header_strategy]
        np.testing.assert_array_equal(second.data, 2)
        np.testing.assert_array_equal(second.uncertainty, 4)
        np.testing.assert_array_equal(second.flags, 3)
        assert dict(second.header) == header2
        if not inplace:
            np.testing.assert_array_equal(first.data, 4)
            np.testing.assert_array_equal(first.uncertainty, 3)
            np.testing.assert_array_equal(first.flags, 1)
            assert dict(first.header) == header1

    def test_reuse_honors_current_configuration(self, monkeypatch):
        arithmetic = ImArith(merge_headers='first')
        frame = FrameData(np.full((2, 3), 4.), unit='m', uncertainty=3.)
        other = QFloat(2, 4, 'm')
        np.testing.assert_array_equal(arithmetic.run(frame, other, '+').uncertainty, 5)
        with monkeypatch.context() as patch:
            patch.setattr(conf, 'FRAMEDATA_DISABLE_UNCERTAINTY', True)
            patch.setattr(conf, 'FRAMEDATA_DISABLE_FLAGS', True)
            result = arithmetic.run(frame, other, '+')
            np.testing.assert_array_equal(result.data, 6)
            assert result.uncertainty is result.flags is None
        np.testing.assert_array_equal(arithmetic.run(frame, other, '+').uncertainty, 5)
        np.testing.assert_array_equal(frame.uncertainty, 3)

    @pytest.mark.parametrize('entry', ['function', 'class'])
    @pytest.mark.parametrize('option', [None, 'FRAMEDATA_DISABLE_UNCERTAINTY'])
    def test_broadcasting_with_exact_operand(self, monkeypatch, entry, option):
        first = FrameData([[4.], [9.]], uncertainty=[[3.], [4.]], unit='m')
        second = np.array([[100., 200., 300.]]) * units.cm
        if option:
            monkeypatch.setattr(conf, option, True)
        result = self.run(entry, first, second, '+')
        np.testing.assert_array_equal(result.data, [[5, 6, 7], [10, 11, 12]])
        assert result.unit == units.m
        if option:
            assert result.uncertainty is None
        else:
            np.testing.assert_array_equal(result.uncertainty, [[3, 3, 3], [4, 4, 4]])
        np.testing.assert_array_equal(first.data, [[4], [9]])
        np.testing.assert_array_equal(second.value, [[100, 200, 300]])

    @pytest.mark.parametrize('entry', ['function', 'class'])
    @pytest.mark.parametrize('option', [None, 'FRAMEDATA_DISABLE_UNCERTAINTY'])
    @pytest.mark.parametrize('other', [units.m, 'm'])
    def test_unit_operands(self, monkeypatch, entry, option, other):
        frame = FrameData(np.full((2, 3), 4.), uncertainty=3., unit='m')
        if option:
            monkeypatch.setattr(conf, option, True)
        result = self.run(entry, frame, other, '*')
        np.testing.assert_array_equal(result.data, 4)
        assert result.unit == units.m**2
        if option:
            assert result.uncertainty is None
        else:
            np.testing.assert_array_equal(result.uncertainty, 3)

    @pytest.mark.parametrize('entry', ['function', 'class'])
    @pytest.mark.parametrize('option', [None, 'FRAMEDATA_DISABLE_UNCERTAINTY'])
    @pytest.mark.parametrize('other', [None, [[1, None]], [[1j]], np.array([])])
    def test_invalid_nominal_values(self, monkeypatch, entry, option, other):
        if option:
            monkeypatch.setattr(conf, option, True)
        with pytest.raises(TypeError):
            self.run(entry, FrameData(np.ones((2, 3))), other, '+')

    @pytest.mark.parametrize('option', [None, 'FRAMEDATA_DISABLE_UNCERTAINTY'])
    def test_unsupported_errors_are_ignored_only_when_disabled(self, monkeypatch, option):
        source = CCDData(np.ones((2, 3)), unit='m',
                         uncertainty=UnknownUncertainty(np.ones((2, 3))))
        if option:
            monkeypatch.setattr(conf, option, True)
            assert imarith(source, 1 * units.m, '+').uncertainty is None
        else:
            with pytest.raises(TypeError):
                imarith(source, 1 * units.m, '+')

    @pytest.mark.parametrize('entry', ['function', 'class'])
    @pytest.mark.parametrize('memmap', [False, True])
    @pytest.mark.parametrize('inplace', [False, True])
    def test_aliasing_and_memmaps(self, tmp_path, entry, memmap, inplace):
        frame = FrameData(np.full((2, 3), 4.), uncertainty=3., unit='m',
                          flags=np.ones((2, 3), dtype='u1'), meta={'OBJECT': 'test'},
                          cache_folder=str(tmp_path / 'frame'), use_memmap_backend=memmap)
        result = self.run(entry, frame, frame, '+', inplace=inplace, merge_headers='first')
        assert (result is frame) is inplace
        np.testing.assert_array_equal(result.data, 8)
        np.testing.assert_allclose(result.uncertainty, np.sqrt(18))
        np.testing.assert_array_equal(result.flags, 1)
        assert result.header['OBJECT'] == 'test'
        if inplace:
            assert isinstance(result.data, np.memmap) is memmap
        else:
            np.testing.assert_array_equal(frame.data, 4)
            np.testing.assert_array_equal(frame.uncertainty, 3)
        frame.disable_memmap()

    @pytest.mark.parametrize('entry', ['function', 'class'])
    @pytest.mark.parametrize('failure', ['units', 'shape', 'flags', 'headers', 'keys'])
    def test_failures_leave_inputs_and_handlers_intact(self, entry, failure):
        from astropop.logger import logger
        frame = FrameData(np.full((2, 3), 4.), uncertainty=3., unit='m',
                          flags=np.ones((2, 3), dtype='u1'), meta={'OBJECT': 'test'})
        other = {'units': 2 * units.s, 'shape': np.ones((5, 5)) * units.m}
        options = {'flags': {'merge_flags': 'invalid'},
                   'headers': {'merge_headers': 'invalid'},
                   'keys': {'merge_headers': 'selected_keys'}}
        handlers = list(logger.handlers)
        with pytest.raises((ValueError, UnitsError)):
            self.run(entry, frame, other.get(failure, 2 * units.m), '+', inplace=True,
                     **options.get(failure, {}))
        assert logger.handlers == handlers
        np.testing.assert_array_equal(frame.data, 4)
        np.testing.assert_array_equal(frame.uncertainty, 3)
        np.testing.assert_array_equal(frame.flags, 1)
        assert frame.unit == units.m
        assert frame.header['OBJECT'] == 'test'

    @pytest.mark.parametrize('entry', ['function', 'class'])
    @pytest.mark.parametrize('inplace', [False, True])
    def test_operation_history_and_handler_cleanup(self, caplog, entry, inplace):
        from astropop.logger import logger
        frame = FrameData(np.ones((2, 3)))
        frame.history = 'original'
        handlers = list(logger.handlers)
        with caplog.at_level('DEBUG', logger=logger.name):
            result = self.run(entry, frame, 2, '*', inplace=inplace)
        assert any(message.startswith('Operation *') for message in result.history)
        assert ('original' in result.history) is inplace
        if not inplace:
            assert frame.history == ['original']
        assert logger.handlers == handlers

    def test_wrapper_forwards_settings_and_ignores_extra_keywords(self, monkeypatch):
        import importlib
        module = importlib.import_module('astropop.image.imarith')
        calls = []
        result = object()

        class Arithmetic:
            def __init__(self, **options):
                calls.append(options)

            def run(self, *args, **options):
                calls.append((args, options))
                return result

        monkeypatch.setattr(module, 'ImArith', Arithmetic)
        assert imarith(4, 2, '*', True, 'and', 'selected_keys',
                       selected_keys=['OBJECT'], unused=True) is result
        assert calls == [dict(merge_flags='and', merge_headers='selected_keys',
                              selected_keys=['OBJECT']),
                         ((4, 2, '*'), {'inplace': True})]


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
        exp_flag = np.zeros(SHAPE, dtype=np.uint8)

        if isinstance(frame2, FrameData):
            # If frame2 is qfloat, quantity or a number, it don't have mask
            flag2 = np.zeros(SHAPE, dtype=np.uint8)
            flag2[3, 3] = 1
            flag2[2, 2] = 6
            frame2.flags = flag2

            if handle_flags == 'and':
                exp_flag[2, 2] = 2
            elif handle_flags == 'or':
                exp_flag[1, 1] = 1
                exp_flag[2, 2] = 6
                exp_flag[3, 3] = 1
        elif handle_flags == 'or':
            exp_flag[1, 1] = 1
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


@pytest.mark.parametrize('inplace', [True, False])
@pars
def test_imarith_ops_frames(op, vs, inplace):
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

    res = imarith(frame1, frame2, op, inplace=inplace)

    assert_almost_equal(res.data, exp_res.data)
    assert_almost_equal(res.uncertainty, exp_res.uncertainty)

    if inplace:
        assert_is(res, frame1)
    else:
        assert_is_not(res, frame1)
