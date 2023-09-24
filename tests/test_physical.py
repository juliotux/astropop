# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from astropop.framedata import FrameData
from astropop.math._deriv import propagate_1, propagate_2, \
                                 numerical_derivative
from astropop.math.physical import QFloat, qfloat, units, \
                                   same_unit, UnitsError, \
                                   equal_within_errors, \
                                   convert_to_qfloat

from astropop.testing import *

# pylint: disable=no-member, pointless-statement


class Test_Derivatives:
    def test_propagate_1_error_unkown(self):
        with pytest.raises(ValueError,
                           match='func test not in derivatives.'):
            propagate_1('test', 1, 1, 1)

    def test_propagate_1_error_not1var(self):
        with pytest.raises(ValueError,
                           match='func div is not a 1 variable function.'):
            propagate_1('div', 1, 1, 1)

    def test_propagate_1_zerodivision(self):
        assert_true(np.isnan(propagate_1('arccos', 0, 2, 0.1)))
        assert_true(np.isnan(propagate_1('arccos',
                                         np.array([0, 0]),
                                         np.array([2, 3]),
                                         np.array([0.1, 0.1]))).all())

    def test_propagate_2_error_unkown(self):
        with pytest.raises(ValueError,
                           match='func test not in derivatives.'):
            propagate_2('test', 1, 1, 1, 1, 1)

    def test_propagate_2_error_not2var(self):
        with pytest.raises(ValueError,
                           match='func cos is not a 2 variable function.'):
            propagate_2('cos', 1, 1, 1, 1, 1)

    def test_propagate_2_zerodivision(self):
        r = propagate_2('div', 1000, 1, 0, 0.1, 0.1)
        assert_true(np.isnan(r))

    def test_propagate_2_zerodivision_array(self):
        r = propagate_2('div',
                        np.array([1000, 1000]),
                        np.ones(2), np.zeros(2),
                        np.array([0.1, 0.1]),
                        np.array([0.1, 0.1]))
        # numpy do not raise zerodivision. instead, set to inf
        # assert_true(np.isnan(r).all())
        assert_true(np.isinf(r).all())

    def test_numerical_derivatives_not_callable_error(self):
        # test raise error for non-callabel functions
        with pytest.raises(TypeError,
                           match='function test not callable.'):
            numerical_derivative('test', 1)

    def test_numerical_derivatives_change_kwargs(self):
        # change kwargs if arg_ref is string
        def test_func(a):
            return a

        deriv = numerical_derivative(test_func, arg_ref='a')
        assert_equal(deriv(a=1), 1)


class Test_QFloat_UnitsHandling:
    def test_qfloat_same_unit(self):
        qf1 = QFloat(1.0, 0.1, 'm')
        qf2 = QFloat(200, 10, 'cm')
        qf3 = QFloat(120, 0.3, 's')
        qf4 = QFloat(2, 0.01, 'min')

        # must be ok, without any convertion
        qf_1, qf_2 = same_unit(qf1, qf1)
        assert_equal(qf_1.nominal, 1.0)
        assert_equal(qf_1.uncertainty, 0.1)
        assert_equal(qf_1.unit, units.m)
        assert_equal(qf_2.nominal, 1.0)
        assert_equal(qf_2.uncertainty, 0.1)
        assert_equal(qf_2.unit, units.m)

        # Doing conversion
        qf_1, qf_2 = same_unit(qf1, qf2)
        assert_equal(qf_1.unit, units.m)
        assert_equal(qf_1.nominal, 1.0)
        assert_equal(qf_1.uncertainty, 0.1)
        assert_equal(qf_2.unit, units.m)
        assert_equal(qf_2.nominal, 2.0)
        assert_equal(qf_2.uncertainty, 0.1)

        # Inverse Conversion
        qf_2, qf_1 = same_unit(qf2, qf1)
        assert_equal(qf_1.unit, units.cm)
        assert_equal(qf_1.nominal, 100)
        assert_equal(qf_1.uncertainty, 10)
        assert_equal(qf_2.unit, units.cm)
        assert_equal(qf_2.nominal, 200)
        assert_equal(qf_2.uncertainty, 10)

        # Raise conversion
        with pytest.raises(UnitsError):
            same_unit(qf3, qf1, self.test_qfloat_same_unit)

        # non-prefix conversion
        qf_3, qf_4 = same_unit(qf3, qf4)
        assert_equal(qf_3.nominal, 120)
        assert_equal(qf_3.uncertainty, 0.3)
        assert_equal(qf_3.unit, units.s)
        assert_equal(qf_4.nominal, 120)
        assert_equal(qf_4.uncertainty, 0.6)
        assert_equal(qf_4.unit, units.s)

        # Inverse non-prefix conversion
        qf_4, qf_3 = same_unit(qf4, qf3)
        assert_equal(qf_3.nominal, 2)
        assert_equal(qf_3.uncertainty, 0.005)
        assert_equal(qf_3.unit, units.minute)
        assert_equal(qf_4.nominal, 2)
        assert_equal(qf_4.uncertainty, 0.01)
        assert_equal(qf_4.unit, units.minute)

    def test_qfloat_same_unit_array(self):
        # like for single values, everything must run as arrays
        qf1 = QFloat(1.0, 0.1, 'm')
        qf2 = QFloat(np.arange(0, 2000, 100), np.arange(0, 100, 5), 'cm')
        qf3 = QFloat([120, 240], [0.3, 0.6], 's')
        qf4 = QFloat([2, 10], [0.01, 0.02], 'min')

        # Simplre conversion
        qf_1, qf_2 = same_unit(qf1, qf2, self.test_qfloat_same_unit_array)
        assert_equal(qf_1.nominal, 1.0)
        assert_equal(qf_1.uncertainty, 0.1)
        assert_equal(qf_1.unit, units.m)
        assert_almost_equal(qf_2.nominal, np.arange(0, 20, 1))
        assert_almost_equal(qf_2.std_dev, np.arange(0, 1, 0.05))
        assert_equal(qf_2.unit, units.m)

        # inverse conversion
        qf_2, qf_1 = same_unit(qf2, qf1, self.test_qfloat_same_unit_array)
        assert_equal(qf_1.nominal, 100)
        assert_equal(qf_1.uncertainty, 10)
        assert_equal(qf_1.unit, units.cm)
        assert_almost_equal(qf_2.nominal, np.arange(0, 2000, 100))
        assert_almost_equal(qf_2.std_dev, np.arange(0, 100, 5))
        assert_equal(qf_2.unit, units.cm)

        # incompatible
        with pytest.raises(UnitsError):
            same_unit(qf3, qf1, self.test_qfloat_same_unit)

        qf_3, qf_4 = same_unit(qf3, qf4, self.test_qfloat_same_unit_array)
        assert_almost_equal(qf_3.nominal, [120, 240])
        assert_almost_equal(qf_3.std_dev, [0.3, 0.6])
        assert_equal(qf_3.unit, units.s)
        assert_almost_equal(qf_4.nominal, [120, 600])
        assert_almost_equal(qf_4.std_dev, [0.6, 1.2])
        assert_equal(qf_4.unit, units.s)


class Test_QFloat_InitAndSet:
    @pytest.mark.parametrize('value, uncertainty, unit', [(1.0, 0.1, 'm'),
                                                          (200, None, None),
                                                          (120, 0.3, None),
                                                          (2.0, None, 'm'),
                                                          (np.arange(10), np.arange(10)*0.1, 'm'),
                                                          ([1, 2, 3], [0.1, 0.2, 0.3], None),
                                                          ([1, 2, 3], [0.1, 0.2, 0.3], 'm')])
    def test_qfloat_init_simple(self, value, uncertainty, unit):
        qf = QFloat(value, uncertainty, unit)
        assert_equal(qf.nominal, value)
        if uncertainty is None:
            assert_equal(qf.uncertainty, np.zeros_like(value))
        else:
            assert_equal(qf.uncertainty, uncertainty)
        if unit is None:
            assert_equal(qf.unit, units.dimensionless_unscaled)
        else:
            assert_equal(qf.unit, unit)

    @pytest.mark.parametrize('value', ['test', complex(1, 2), None, b'0\x00',
                                       [0, 'test', 2], [0, None, 0]])
    def test_qfloat_init_notreal(self, value):
        with pytest.raises(TypeError):
            QFloat(value)

    def test_qfloat_init_notreal_uncert(self):
        with pytest.raises(TypeError):
            QFloat(1, 'test')

        with pytest.raises(TypeError):
            QFloat(1, complex(1, 2))

        with pytest.raises(TypeError):
            QFloat([1, 2, 3], [1, 'Test', 2])

        qf = QFloat([1, 2, 3], [1, None, 3])
        assert_equal(qf.nominal, [1, 2, 3])
        assert_equal(qf.uncertainty, [1, 0, 3])

        with pytest.raises(TypeError):
            qf.uncertainty = [1, 'Test', 3]

    def test_qfloat_init_qfloat(self):
        qf1 = QFloat(1.0, 0.1, 'm')
        qf2 = QFloat(qf1)
        assert_equal(qf2.nominal, 1.0)
        assert_equal(qf2.uncertainty, 0.1)
        assert_equal(qf2.unit, units.m)
        assert_is_not(qf2, qf1)

    def test_qfloat_init_qfloat_conflicts(self):
        qf = QFloat(1.0, 0.1, 'm')
        with pytest.raises(ValueError):
            QFloat(qf, unit='m')
        with pytest.raises(ValueError):
            QFloat(qf, uncertainty=0.1)

    def test_qfloat_set_sig_digits(self):
        qf = QFloat(1.0, 0.1, 'm')
        qf.sig_digits = 2
        assert_equal(qf.nominal, 1.0)
        assert_equal(qf.uncertainty, 0.1)
        assert_equal(qf.unit, units.m)
        assert_equal(qf.sig_digits, 2)

    def test_qfloat_set_sig_digits_error(self):
        qf = QFloat(1.0, 0.1, 'm')
        with pytest.raises(TypeError,
                           match='sig_digits must be a real number.'):
            qf.sig_digits = 'k'


class Test_QFloat_Repr:
    def test_qfloat_repr_default(self):
        for qf, rep in [(QFloat(1.0, 0.1, 'm'), '1.0+-0.1 m'),
                        (QFloat(1.0, 0.01, 'm'), '1.00+-0.01 m'),
                        (QFloat(1.00, 0.01, 'm'), '1.00+-0.01 m'),
                        (QFloat(1.00, 0.012, 'm'), '1.00+-0.01 m'),
                        (QFloat(1.00, 0.016, 'm'), '1.00+-0.02 m'),
                        (QFloat(1.002, 0.016, 'm'), '1.00+-0.02 m'),
                        (QFloat(1.006, 0.016, 'm'), '1.01+-0.02 m'),
                        (QFloat(1.000005, 0.01, 'm'), '1.00+-0.01 m'),
                        (QFloat(1.000005, 0.001, 'm'), '1.000+-0.001 m'),
                        (QFloat(98, 0.98, 'm'), '98+-1 m'),
                        (QFloat(98.7, 0.98, 'm'), '99+-1 m'),
                        (QFloat(9999, 1, 'm'), '9999+-1 m'),
                        (QFloat(9999, 10, 'm'), '10000+-10 m')]:
            i = hex(id(qf))
            assert_equal(repr(qf), f'<QFloat at {i}>\n{rep}')

    def test_qfloat_repr_sig_digits(self):
        for qf, rep in [(QFloat(1.0, 0.1, 'm'), '1.00+-0.10 m'),
                        (QFloat(1.0, 0.01, 'm'), '1.000+-0.010 m'),
                        (QFloat(1.00, 0.01, 'm'), '1.000+-0.010 m'),
                        (QFloat(1.00, 0.012, 'm'), '1.000+-0.012 m'),
                        (QFloat(1.00, 0.016, 'm'), '1.000+-0.016 m'),
                        (QFloat(1.002, 0.016, 'm'), '1.002+-0.016 m'),
                        (QFloat(1.006, 0.016, 'm'), '1.006+-0.016 m'),
                        (QFloat(1.000005, 0.01, 'm'), '1.000+-0.010 m'),
                        (QFloat(1.000005, 0.001, 'm'), '1.0000+-0.0010 m'),
                        (QFloat(9999, 1, 'm'), '9999.0+-1.0 m'),
                        (QFloat(9999, 10, 'm'), '9999+-10 m')]:
            qf.sig_digits = 2
            i = hex(id(qf))
            assert_equal(repr(qf), f'<QFloat at {i}>\n{rep}')

    def test_qfloat_repr_array(self):
        n = np.array([1, 1, 1, 1.01, 1.001, 1.006])
        s = np.array([0.1, 0.01, 1.0, 0.01, 0.01, 0.01])
        qf = QFloat(n, s, 'm')
        i = hex(id(qf))
        assert_equal(repr(qf),
                     f'<QFloat at {i}>\n'
                     '[1.0+-0.1, 1.00+-0.01, 1+-1, 1.01+-0.01, '
                     '1.00+-0.01, 1.01+-0.01] unit=m')

        # 1D very large array
        qf = QFloat(np.arange(1, 601), np.arange(1, 601)*0.01, 'm')
        i = hex(id(qf))
        assert_equal(repr(qf),
                     f'<QFloat at {i}>\n'
                     '[1.00+-0.01, 2.00+-0.02, 3.00+-0.03, ..., 598+-6, '
                     '599+-6, 600+-6] unit=m')

        # very large 2D array
        n = np.arange(1, 10001).reshape((100, 100))
        s = n*0.01
        qf = QFloat(n, s, 's')
        i = hex(id(qf))
        assert_equal(repr(qf),
                     f'<QFloat at {i}>\n'
                     '[[1.00+-0.01, 2.00+-0.02, 3.00+-0.03, ..., 98+-1, 99+-1,'
                     ' 100+-1],\n [101+-1, 102+-1, 103+-1, ..., 198+-2, '
                     '199+-2, 200+-2],\n [201+-2, 202+-2, 203+-2, ..., '
                     '298+-3, 299+-3, 300+-3],\n ...,\n [9700+-100, 9700+-100,'
                     ' 9700+-100, ..., 9800+-100, 9800+-100, 9800+-100],\n'
                     ' [9800+-100, 9800+-100, 9800+-100, ..., 9900+-100, '
                     '9900+-100, 9900+-100],\n [9900+-100, 9900+-100, '
                     '9900+-100, ..., 10000+-100, 10000+-100,\n  10000+-100]] '
                     'unit=s')

    def test_qfloat_repr_array_sig_digits(self):
        n = np.array([1, 1, 1, 1.01, 1.001, 1.006])
        s = np.array([0.12, 0.018, 1.08, 0.0126, 0.01, 0.01])
        qf = QFloat(n, s, 'm')
        i = hex(id(qf))
        qf.sig_digits = 2
        assert_equal(repr(qf),
                     f'<QFloat at {i}>\n'
                     '[1.00+-0.12, 1.000+-0.018, 1.0+-1.1, 1.010+-0.013, '
                     '1.001+-0.010,\n 1.006+-0.010] unit=m')

    def test_qfloat_str_no_unit(self):
        qf = QFloat(1.0, 0.1)
        # must not have space after uncertainty
        assert_equal(str(qf), '1.0+-0.1')

    def test_qfloat_str(self):
        qf = QFloat(1.0, 0.1, 'm')
        assert_equal(str(qf), '1.0+-0.1 m')

    def test_qfloat_str_array(self):
        n = np.array([1, 1, 1, 1.01, 1.001, 1.006])
        s = np.array([0.1, 0.01, 1.0, 0.01, 0.01, 0.01])
        qf = QFloat(n, s, 'm')
        assert_equal(str(qf),
                     '[1.0+-0.1, 1.00+-0.01, 1+-1, 1.01+-0.01, 1.00+-0.01, '
                     '1.01+-0.01] unit=m')

    def test_qfloat_str_array_no_unit(self):
        n = np.array([1, 1, 1, 1.01, 1.001, 1.006])
        s = np.array([0.1, 0.01, 1.0, 0.01, 0.01, 0.01])
        qf = QFloat(n, s)
        assert_equal(str(qf),
                     '[1.0+-0.1, 1.00+-0.01, 1+-1, 1.01+-0.01, 1.00+-0.01, '
                     '1.01+-0.01]')


class Test_QFloat_Operators:
    def test_qfloat_properties_getset(self):
        # access all important properties
        qf = QFloat(5.0, 0.025, 'm')
        assert_equal(qf.uncertainty, 0.025)
        qf.uncertainty = 0.10
        assert_equal(qf.uncertainty, 0.10)
        assert_equal(qf.std_dev, 0.10)
        qf.uncertainty = 0.05
        assert_equal(qf.std_dev, 0.05)

        # setting nominal resets the uncertainty
        assert_equal(qf.nominal, 5.0)

        with pytest.raises(TypeError):
            qf.nominal = None

        assert_equal(qf.nominal, 5.0)
        qf.nominal = 10.0
        assert_equal(qf.nominal, 10.0)
        assert_equal(qf.std_dev, 0.0)
        assert_equal(qf.uncertainty, 0.0)

        # with arrays
        qf = QFloat([1, 2, 3], [0.1, 0.2, 0.3], 'm')
        assert_almost_equal(qf.uncertainty, [0.1, 0.2, 0.3])
        qf.uncertainty = [0.4, 0.5, 0.6]
        assert_almost_equal(qf.uncertainty, [0.4, 0.5, 0.6])
        assert_almost_equal(qf.std_dev, [0.4, 0.5, 0.6])
        qf.std_dev = [0.1, 0.2, 0.3]
        assert_almost_equal(qf.std_dev, [0.1, 0.2, 0.3])
        assert_almost_equal(qf.nominal, [1, 2, 3])

        with pytest.raises(TypeError):
            qf.nominal = None

        assert_almost_equal(qf.nominal, [1, 2, 3])
        qf.nominal = [4, 5, 6]
        assert_almost_equal(qf.nominal, [4, 5, 6])
        assert_almost_equal(qf.std_dev, [0, 0, 0])
        assert_almost_equal(qf.uncertainty, [0, 0, 0])

        # test value readonly property
        assert_equal(qf.value, [4, 5, 6])
        with pytest.raises(AttributeError):
            qf.value = 0

    def test_qfloat_set_nominal(self):
        qf = QFloat(5.0, 0.025, 'm')
        qf.nominal = 10.0
        assert_equal(qf.nominal, 10.0)
        assert_equal(qf.std_dev, 0.0)
        assert_equal(qf.uncertainty, 0.0)

        # with arrays
        qf = QFloat([1, 2, 3], [0.1, 0.2, 0.3], 'm')
        qf.nominal = [4, 5, 6]
        assert_almost_equal(qf.nominal, [4, 5, 6])
        assert_almost_equal(qf.std_dev, [0, 0, 0])
        assert_almost_equal(qf.uncertainty, [0, 0, 0])
        assert_equal(qf.unit, units.m)

        # tuple for (nominal, uncertainty)
        qf = QFloat(5.0, 0.025, 'm')
        qf.nominal = (10.0, 0.1)
        assert_equal(qf.nominal, 10.0)
        assert_equal(qf.uncertainty, 0.1)
        assert_equal(qf.unit, units.m)

        # tuple for (nominal, uncertainty, unit)
        qf = QFloat(5.0, 0.025, 'm')
        qf.nominal = (10.0, 0.1, 's')
        assert_equal(qf.nominal, 10.0)
        assert_equal(qf.uncertainty, 0.1)
        assert_equal(qf.unit, units.s)

        # raise non real
        with pytest.raises(TypeError):
            qf.nominal = None
        with pytest.raises(TypeError):
            qf.nominal = [1, 'test', 2]

    def test_qfloat_set_uncertainty(self):
        qf = QFloat(5.0, 0.025, 'm')
        qf.uncertainty = 0.1
        assert_equal(qf.uncertainty, 0.1)
        assert_equal(qf.std_dev, 0.1)

        # with arrays
        qf = QFloat([1, 2, 3], [0.1, 0.2, 0.3], 'm')
        qf.uncertainty = [0.4, 0.5, 0.6]
        assert_almost_equal(qf.uncertainty, [0.4, 0.5, 0.6])
        assert_almost_equal(qf.std_dev, [0.4, 0.5, 0.6])
        qf.uncertainty = None
        assert_almost_equal(qf.uncertainty, [0, 0, 0])

        # raise non real
        with pytest.raises(TypeError):
            qf.uncertainty = 'test'
        with pytest.raises(TypeError):
            qf.uncertainty = [1, 'test', 2]

        # raise invalid shape
        with pytest.raises(ValueError):
            qf.uncertainty = [1, 2, 3, 4]
        with pytest.raises(ValueError):
            qf.uncertainty = 1

    def test_qfloat_properties_reset(self):
        qf = QFloat(5.0, 0.025, 'm')
        i = id(qf)
        qf.reset(12, 0.2, 's')
        assert_equal(i, id(qf))
        assert_equal(qf, QFloat(12, 0.2, 's'))

        qf.reset([1, 2, 3], [0.1, 0.2, 0.3], 'm')
        assert_equal(i, id(qf))
        assert_equal(qf, QFloat([1, 2, 3], [0.1, 0.2, 0.3], 'm'))

    def test_qfloat_creation(self):
        def _create(*args, **kwargs):
            assert_is_instance(qfloat(*args, **kwargs), QFloat)

        def _raises(*args, **kwargs):
            with pytest.raises(Exception):
                qfloat(*args, **kwargs)

        # Simple scalar
        _create(1.0)
        _create(1.0, 0.1)
        _create(1.0, 0.1, 'm')

        # Arrays
        _create(np.ones(10))
        _create(np.ones(10), np.ones(10))
        _create(np.ones(10), np.ones(10), 'm')

        # Fails
        _raises(np.ones(10), 0.1)
        _raises(10.0, np.ones(10))
        _raises(uncertainty=0.1)
        _raises(unit='m')
        _raises(uncertainty=0.1, unit='m')

    def test_qfloat_always_positive_uncertainty(self):
        qf1 = QFloat(1.0, 0.1, 'm')
        assert_equal(qf1.uncertainty, 0.1)

        qf2 = QFloat(1.0, -0.1, 'm')
        assert_equal(qf2.uncertainty, 0.1)

        qf3 = QFloat(np.ones(10), np.ones(10)*0.1, 'm')
        assert_equal(qf3.uncertainty, np.ones(10)*0.1)

        qf4 = QFloat(np.ones(10), -np.ones(10)*0.1, 'm')
        assert_equal(qf4.uncertainty, np.ones(10)*0.1)

    @pytest.mark.parametrize('value,expect', [(QFloat(1.0, 0.1, 'm'), QFloat(1.0, 0.1, 'm')),
                                              (1, QFloat(1.0, 0, None)),
                                              (np.array([1, 2, 3]), QFloat([1, 2, 3], unit=None)),
                                              ('string', 'raise'),
                                              (None, 'raise'),
                                              (UnitsError, 'raise'),
                                              (1.0*units.m, QFloat(1.0, 0.0, 'm')),
                                              (FrameData([[1.0]], 'm', 'f8', 0.1), QFloat([[1.0]], [[0.1]], 'm'))])
    def test_qfloat_converttoqfloat(self, value, expect):
        if expect == 'raise':
            with pytest.raises(Exception):
                convert_to_qfloat(value)
        else:
            conv = convert_to_qfloat(value)
            assert_equal(conv.nominal, expect.nominal)
            assert_equal(conv.uncertainty, expect.uncertainty)
            assert_equal(conv.unit, expect.unit)

    def test_qfloat_unit_conversion(self):
        qf1 = QFloat(1.0, 0.01, 'm')

        # test converting with string
        qf2 = qf1 << 'cm'
        assert_equal(qf2.nominal, 100)
        assert_equal(qf2.uncertainty, 1)
        assert_equal(qf2.unit, units.cm)

        # test converting using instance
        qf3 = qf1 << units.cm
        assert_equal(qf3.nominal, 100)
        assert_equal(qf3.uncertainty, 1)
        assert_equal(qf3.unit, units.cm)

        # But qf1 must stay the same
        assert_equal(qf1.nominal, 1.0)
        assert_equal(qf1.uncertainty, 0.01)
        assert_equal(qf1.unit, units.m)

        with pytest.raises(Exception):
            # None must fail
            qf1 << None

        with pytest.raises(Exception):
            # incompatible must fail
            qf1 << units.s

        # and this must work with arrays
        qf1 = QFloat([1, 2], [0.1, 0.2], unit='m')
        qf2 = qf1 << 'km'
        assert_equal(qf2.nominal, [0.001, 0.002])
        assert_equal(qf2.uncertainty, [0.0001, 0.0002])
        assert_equal(qf2.unit, units.km)

    def test_qfloat_unit_conversion_to(self):
        qf1 = QFloat(1.0, 0.01, 'm')

        # test converting with string
        qf2 = qf1.to('cm')
        assert_equal(qf2.nominal, 100)
        assert_equal(qf2.uncertainty, 1)
        assert_equal(qf2.unit, units.cm)

        # test converting using instance
        qf3 = qf1.to(units.cm)
        assert_equal(qf3.nominal, 100)
        assert_equal(qf3.uncertainty, 1)
        assert_equal(qf3.unit, units.cm)

        # But qf1 must stay the same
        assert_equal(qf1.nominal, 1.0)
        assert_equal(qf1.uncertainty, 0.01)
        assert_equal(qf1.unit, units.m)

        # inline: now qf1 must change
        i = id(qf1)
        qf1 <<= 'cm'
        assert_equal(qf1.nominal, 100)
        assert_equal(qf1.uncertainty, 1)
        assert_equal(qf1.unit, units.cm)
        assert_equal(id(qf1), i)

        with pytest.raises(Exception):
            # None must fail
            qf1.to(None)

        with pytest.raises(Exception):
            # incompatible must fail
            qf1.to(units.s)

        # and this must work with arrays
        qf1 = QFloat([1, 2], [0.1, 0.2], unit='m')
        qf2 = qf1.to('km')
        assert_equal(qf2.nominal, [0.001, 0.002])
        assert_equal(qf2.uncertainty, [0.0001, 0.0002])
        assert_equal(qf2.unit, units.km)

    def test_qfloat_getitem(self):
        # simple array
        qf = QFloat([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5], 's')

        qf1 = qf[0]
        assert_equal(qf1.nominal, 1)
        assert_equal(qf1.uncertainty, 0.1)
        assert_equal(qf1.unit, units.s)

        qf3 = qf[2]
        assert_equal(qf3.nominal, 3)
        assert_equal(qf3.uncertainty, 0.3)
        assert_equal(qf3.unit, units.s)

        qf4 = qf[-1]
        assert_equal(qf4.nominal, 5)
        assert_equal(qf4.uncertainty, 0.5)
        assert_equal(qf4.unit, units.s)

        qf5 = qf[1:4]
        assert_equal(qf5.nominal, [2, 3, 4])
        assert_equal(qf5.uncertainty, [0.2, 0.3, 0.4])
        assert_equal(qf5.unit, units.s)

        with pytest.raises(IndexError):
            qf[10]

        # 2D array
        qf = QFloat(np.arange(1, 17, 1).reshape((4, 4)),
                    np.arange(1, 17, 1).reshape((4, 4))*0.01, 'm')

        qfrow = qf[0]
        assert_equal(qfrow.nominal, [1, 2, 3, 4])
        assert_equal(qfrow.uncertainty, [0.01, 0.02, 0.03, 0.04])
        assert_equal(qfrow.unit, units.m)

        qfcol = qf[:, 1]
        assert_equal(qfcol.nominal, [2, 6, 10, 14])
        assert_equal(qfcol.uncertainty, [0.02, 0.06, 0.1, 0.14])
        assert_equal(qfcol.unit, units.m)

        qf0 = qf[0, 0]
        assert_equal(qf0.nominal, 1)
        assert_equal(qf0.uncertainty, 0.01)
        assert_equal(qf0.unit, units.m)

        qf1 = qf[-1, -1]
        assert_equal(qf1.nominal, 16)
        assert_equal(qf1.uncertainty, 0.16)
        assert_equal(qf1.unit, units.m)

        qfs = qf[2:, 1:3]
        assert_equal(qfs.nominal, [[10, 11], [14, 15]])
        assert_equal(qfs.uncertainty, [[0.10, 0.11], [0.14, 0.15]])
        assert_equal(qfs.unit, units.m)

        with pytest.raises(IndexError):
            qf[10]

        with pytest.raises(IndexError):
            qf[0, 10]

        # Not iterable
        qf = QFloat(10, 0.1, 'm')
        with pytest.raises(TypeError):
            qf[0]

    def test_qfloat_setitem(self):
        # simple array
        qf = QFloat([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5], 's')
        qf[0] = QFloat(10, 0.5, 's')
        qf[-1] = QFloat(-10, unit='min')
        qf[2:4] = QFloat(1, 0.3, 's')
        assert_equal(qf.nominal, [10, 2, 1, 1, -600])
        assert_equal(qf.uncertainty, [0.5, 0.2, 0.3, 0.3, 0])
        assert_equal(qf.unit, units.s)

        with pytest.raises(IndexError):
            qf[5] = QFloat(10, 0.1, 's')
        with pytest.raises(UnitsError):
            qf[0] = 10
        with pytest.raises(UnitsError):
            qf[0] = QFloat(10, 0.1, 'm')

        # 2D array
        qf = QFloat(np.arange(1, 17, 1).reshape((4, 4)),
                    np.arange(1, 17, 1).reshape((4, 4))*0.01,
                    'm')
        qf[0] = QFloat(np.arange(5, 9, 1), np.arange(0.3, 0.7, 0.1), 'm')
        qf[:, 0] = QFloat(np.arange(1, 5, 1), [0.1, 0.3, 0.9, 0.4], 'm')
        qf[2, 2] = QFloat(1000, 10, 'cm')
        qf[2:, 2:] = QFloat(20, 0.66, 'm')
        assert_equal(qf.nominal, [[1, 6, 7, 8], [2, 6, 7, 8], [3, 10, 20, 20], [4, 14, 20, 20]])
        assert_almost_equal(qf.uncertainty, [[0.1, 0.4, 0.5, 0.6],
                                             [0.3, 0.06, 0.07, 0.08],
                                             [0.9, 0.1, 0.66, 0.66],
                                             [0.4, 0.14, 0.66, 0.66]])
        assert_equal(qf.unit, units.m)

        with pytest.raises(IndexError):
            qf[5, 10] = QFloat(10, 0.1, 'm')
        with pytest.raises(UnitsError):
            qf[0, 0] = 10
        with pytest.raises(UnitsError):
            qf[0, 0] = QFloat(10, 0.1, 's')

    def test_qfloat_len(self):
        with pytest.raises(TypeError):
            len(QFloat(1.0, 0.1, 'm'))
        assert_equal(len(QFloat([1], [0.1], 'm')), 1)
        assert_equal(len(QFloat([2, 3], [1, 2], 'm')), 2)

        # same behavior of numpy
        assert_equal(len(QFloat(np.zeros((10, 5)), np.zeros((10, 5)), 'm')), 10)
        assert_equal(len(QFloat(np.zeros((10, 5)), np.zeros((10, 5)), 'm')[0]), 5)


class Test_QFloat_Comparison:
    def test_qfloat_comparison_equality_same_unit(self):
        # These numbers must not be equal, but equal within errors.
        qf1 = QFloat(1.0, 0.1, 'm')
        qf2 = QFloat(1.05, 0.2, 'm')
        qf3 = QFloat(1.8, 0.1, 'm')
        assert_false(qf1 == qf2)
        assert_true(qf1 != qf2)
        assert_true(qf1 != qf3)
        assert_true(qf2 != qf3)
        assert_true(equal_within_errors(qf1, qf2))
        assert_false(equal_within_errors(qf1, qf3))

    def test_qfloat_comparison_equality_convert_units(self):
        # Units must matter
        qf1 = QFloat(1.0, 0.1, 'm')
        qf3 = QFloat(1.0, 0.1, 'cm')
        qf4 = QFloat(100, 10, 'cm')
        assert_false(qf1 == qf3)
        assert_false(qf1 != qf4)
        assert_true(qf1 != qf3)
        assert_true(qf1 == qf4)
        assert_false(equal_within_errors(qf1, qf3))
        assert_true(equal_within_errors(qf1, qf4))

    def test_qfloat_comparison_equality_incompatible_units(self):
        qf1 = QFloat(1.0, 0.1, 'm')
        qf2 = QFloat(1.0, 0.1, 's')
        assert_false(qf1 == qf2)
        assert_true(qf1 != qf2)
        assert_false(equal_within_errors(qf1, qf2))

    def test_qfloat_comparison_inequality_same_unit(self):
        qf1 = QFloat(1.0, 0.1, 'm')
        qf2 = QFloat(1.0, 0.2, 'm')
        qf3 = QFloat(1.1, 0.1, 'm')
        qf4 = QFloat(1.1, 0.2, 'm')
        qf5 = QFloat(0.9, 0.1, 'm')
        qf6 = QFloat(0.9, 0.5, 'm')

        assert_true(qf1 <= qf2)
        assert_false(qf1 < qf2)
        assert_true(qf1 >= qf2)
        assert_false(qf1 > qf2)

        assert_true(qf1 <= qf3)
        assert_true(qf1 < qf3)
        assert_false(qf1 > qf3)
        assert_false(qf1 >= qf3)

        assert_true(qf1 <= qf4)
        assert_true(qf1 < qf4)
        assert_false(qf1 > qf4)
        assert_false(qf1 >= qf4)

        assert_true(qf1 >= qf5)
        assert_true(qf1 > qf5)
        assert_false(qf1 < qf5)
        assert_false(qf1 <= qf5)

        assert_true(qf1 >= qf6)
        assert_true(qf1 > qf6)
        assert_false(qf1 < qf6)
        assert_false(qf1 <= qf6)

    def test_qfloat_comparison_inequality_convert_unit(self):
        qf1 = QFloat(1.0, 0.1, 'm')
        qf2 = QFloat(200, 0.1, 'cm')
        qf3 = QFloat(0.005, 0.0003, 'km')
        qf4 = QFloat(0.0001, 0.00001, 'km')

        assert_true(qf2 > qf1)
        assert_true(qf2 >= qf1)
        assert_false(qf2 < qf1)
        assert_false(qf2 <= qf1)

        assert_true(qf3 > qf1)
        assert_true(qf3 >= qf1)
        assert_false(qf3 < qf1)
        assert_false(qf3 <= qf1)

        assert_true(qf4 < qf1)
        assert_true(qf4 <= qf1)
        assert_false(qf4 > qf1)
        assert_false(qf4 >= qf1)

    def test_qfloat_comparison_inequality_incompatible_units(self):
        qf1 = QFloat(1.0, 0.1, 'm')
        qf2 = QFloat(1.0, 0.1, 's')

        with pytest.raises(UnitsError):
            qf1 < qf2

        with pytest.raises(UnitsError):
            qf1 <= qf2

        with pytest.raises(UnitsError):
            qf1 > qf2

        with pytest.raises(UnitsError):
            qf1 >= qf2


class Test_QFloat_Add:
    def test_qfloat_math_add_single(self):
        qf1 = QFloat(1.0, 0.01, 'm')
        qf2 = QFloat(50, 2, 'cm')
        qf3 = QFloat(10, 0.1, None)
        qf4 = QFloat(60, 0.001, 's')

        res1 = qf1 + qf2
        assert_equal(res1.nominal, 1.5)
        assert_almost_equal(res1.uncertainty, 0.022360679774)
        assert_equal(res1.unit, units.m)

        # same as above, but in cm
        res2 = qf2 + qf1
        assert_equal(res2.nominal, 150)
        assert_almost_equal(res2.uncertainty, 2.2360679774)
        assert_equal(res2.unit, units.cm)

        # Should fail with incompatible units
        with pytest.raises(UnitsError):
            qf1 + qf3

        with pytest.raises(UnitsError):
            qf1 + qf4

    def test_qfloat_math_add_array(self):
        qf1 = QFloat([1, 2, 3, 4], [0.01, 0.02, 0.01, 0.02], 'm')
        qf2 = QFloat([150, 250, 50, 550], [0.1, 5, 0.4, 2], 'cm')
        qf3 = QFloat(1, 0.01, 'm')
        qf4 = QFloat(10, 0.1, None)
        qf5 = QFloat(60, 0.001, 's')

        res1 = qf1 + qf2
        assert_equal(res1.nominal, [2.5, 4.5, 3.5, 9.5])
        assert_almost_equal(res1.uncertainty, [0.01004987562112089, 0.05385164807134505,
                                               0.010770329614269008, 0.0282842712474619])
        assert_equal(res1.unit, units.m)

        # same as above, but in cm
        res2 = qf2 + qf1
        assert_equal(res2.nominal, [250, 450, 350, 950])
        assert_almost_equal(res2.uncertainty, [1.004987562112089, 5.385164807134505,
                                               1.0770329614269008, 2.82842712474619])
        assert_equal(res2.unit, units.cm)

        # Numpy behavior is to sum arrays with single numbers
        res3 = qf1 + qf3
        assert_equal(res3.nominal, [2.0, 3.0, 4.0, 5.0])
        assert_almost_equal(res3.uncertainty, [0.01414213562373095, 0.022360679774997897,
                                               0.01414213562373095, 0.022360679774997897])
        assert_equal(res3.unit, units.m)

        # So, it should sum numbers with arrays
        res4 = qf3 + qf1
        assert_equal(res4.nominal, [2.0, 3.0, 4.0, 5.0])
        assert_almost_equal(res4.uncertainty, [0.01414213562373095, 0.022360679774997897,
                                               0.01414213562373095, 0.022360679774997897])
        assert_equal(res4.unit, units.m)

        # Should fail with incompatible units
        with pytest.raises(UnitsError):
            qf1 + qf4

        with pytest.raises(UnitsError):
            qf1 + qf5

    def test_qfloat_math_add_with_numbers(self):
        qf1 = QFloat(1, 0.1, 'm')
        qf2 = QFloat(1, 0.1)  # dimensionless
        qf3 = QFloat([2, 3], [0.1, 0.2])
        qf4 = QFloat([2, 3], [0.1, 0.2], 's')

        # Should raise with dimension measurements
        with pytest.raises(UnitsError):
            qf1 + 1
        with pytest.raises(UnitsError):
            1 + qf1

        # but works with dimensionless
        res1 = qf2 + 2
        assert_equal(res1.nominal, 3)
        assert_almost_equal(res1.uncertainty, 0.1)
        assert_equal(res1.unit, units.dimensionless_unscaled)

        # same as above, but inverse
        res2 = 2 + qf2
        assert_equal(res2.nominal, 3)
        assert_almost_equal(res2.uncertainty, 0.1)
        assert_equal(res2.unit, units.dimensionless_unscaled)

        # and with arrays!
        res3 = qf3 + 1
        assert_equal(res3.nominal, [3, 4])
        assert_almost_equal(res3.uncertainty, [0.1, 0.2])
        assert_equal(res3.unit, units.dimensionless_unscaled)

        # and with arrays inverse!
        res4 = 1 + qf3
        assert_equal(res4.nominal, [3, 4])
        assert_almost_equal(res4.uncertainty, [0.1, 0.2])
        assert_equal(res4.unit, units.dimensionless_unscaled)

        # array array
        with pytest.raises(UnitsError):
            qf4 + [1, 2]

        with pytest.raises(UnitsError):
            [1, 2] + qf4

        res5 = qf3 + [1, 2]
        assert_equal(res5.nominal, [3, 5])
        assert_almost_equal(res5.uncertainty, [0.1, 0.2])
        assert_equal(res5.unit, units.dimensionless_unscaled)

        res6 = [1, 2] + qf3
        assert_equal(res6.nominal, [3, 5])
        assert_almost_equal(res6.uncertainty, [0.1, 0.2])
        assert_equal(res6.unit, units.dimensionless_unscaled)

    def test_qfloat_math_add_inline(self):
        # Sum with another qfloat
        qf = QFloat(1.0, 0.1, 'm')
        i = id(qf)
        qf += QFloat(2.0, 0.1, 'm')
        assert_equal(qf.nominal, 3)
        assert_almost_equal(qf.uncertainty, 0.1414213562373095)
        assert_equal(qf.unit, units.m)
        assert_equal(id(qf), i)

        # raise if incompatible
        with pytest.raises(UnitsError):
            qf += QFloat(3, unit='s')
        assert_equal(id(qf), i)

        with pytest.raises(UnitsError):
            qf += 1
        assert_equal(id(qf), i)

        # Arrays
        qf = QFloat([1, 2, 3, 4], [0.01, 0.02, 0.03, 0.04], 'cm')
        i = id(qf)
        qf += QFloat(1, 0.01, 'cm')
        assert_equal(qf.nominal, [2, 3, 4, 5])
        assert_almost_equal(qf.uncertainty, [0.01414213562373095, 0.022360679774997897,
                                             0.03162277660168379, 0.04123105625617661])
        assert_equal(qf.unit, units.cm)
        assert_equal(id(qf), i)

        with pytest.raises(UnitsError):
            qf += [2, 3, 4, 5]
        assert_equal(id(qf), i)


class Test_QFloat_Sub:
    def test_qfloat_math_sub_single(self):
        qf1 = QFloat(1.0, 0.01, 'm')
        qf2 = QFloat(50, 2, 'cm')
        qf3 = QFloat(10, 0.1, None)
        qf4 = QFloat(60, 0.001, 's')

        res1 = qf1 - qf2
        assert_equal(res1.nominal, 0.5)
        assert_almost_equal(res1.uncertainty, 0.022360679774)
        assert_equal(res1.unit, units.m)

        # same as above, but in cm
        res2 = qf2 - qf1
        assert_equal(res2.nominal, -50)
        assert_almost_equal(res2.uncertainty, 2.2360679774)
        assert_equal(res2.unit, units.cm)

        # Should fail with incompatible units
        with pytest.raises(UnitsError):
            qf1 - qf3

        with pytest.raises(UnitsError):
            qf1 - qf4

    def test_qfloat_math_sub_array(self):
        qf1 = QFloat([1, 2, 3, 4], [0.01, 0.02, 0.01, 0.02], 'm')
        qf2 = QFloat([50, 40, 30, 550], [0.1, 5, 0.4, 2], 'cm')
        qf3 = QFloat(1, 0.01, 'm')
        qf4 = QFloat(10, 0.1, None)
        qf5 = QFloat(60, 0.001, 's')

        res1 = qf1 - qf2
        assert_equal(res1.nominal, [0.5, 1.6, 2.7, -1.5])
        assert_almost_equal(res1.uncertainty, [0.01004987562112089, 0.05385164807134505,
                                               0.010770329614269008, 0.0282842712474619])
        assert_equal(res1.unit, units.m)

        # same as above, but in cm
        res2 = qf2 - qf1
        assert_equal(res2.nominal, [-50, -160, -270, 150])
        assert_almost_equal(res2.uncertainty, [1.004987562112089, 5.385164807134505,
                                               1.0770329614269008, 2.82842712474619])
        assert_equal(res2.unit, units.cm)

        # Numpy behavior is to sum arrays with single numbers
        res3 = qf1 - qf3
        assert_equal(res3.nominal, [0.0, 1.0, 2.0, 3.0])
        assert_almost_equal(res3.uncertainty, [0.01414213562373095, 0.022360679774997897,
                                               0.01414213562373095, 0.022360679774997897])
        assert_equal(res3.unit, units.m)

        # So, it should sum numbers with arrays
        res4 = qf3 - qf1
        assert_equal(res4.nominal, [0.0, -1, -2.0, -3.0])
        assert_almost_equal(res4.uncertainty, [0.01414213562373095, 0.022360679774997897,
                                               0.01414213562373095, 0.022360679774997897])
        assert_equal(res4.unit, units.m)

        # Should fail with incompatible units
        with pytest.raises(UnitsError):
            qf1 - qf4

        with pytest.raises(UnitsError):
            qf1 - qf5

    def test_qfloat_math_sub_with_numbers(self):
        qf1 = QFloat(1, 0.1, 'm')
        qf2 = QFloat(1, 0.1)  # dimensionless
        qf3 = QFloat([2, 3], [0.1, 0.2])
        qf4 = QFloat([2, 3], [0.1, 0.2], 's')

        # Should raise with dimension measurements
        with pytest.raises(UnitsError):
            qf1 - 1
        with pytest.raises(UnitsError):
            1 - qf1

        # but works with dimensionless
        res1 = qf2 - 2
        assert_equal(res1.nominal, -1)
        assert_almost_equal(res1.uncertainty, 0.1)
        assert_equal(res1.unit, units.dimensionless_unscaled)

        # same as above, but inverse
        res2 = 2 - qf2
        assert_equal(res2.nominal, 1)
        assert_almost_equal(res2.uncertainty, 0.1)
        assert_equal(res2.unit, units.dimensionless_unscaled)

        # and with arrays!
        res3 = qf3 - 1
        assert_equal(res3.nominal, [1, 2])
        assert_almost_equal(res3.uncertainty, [0.1, 0.2])
        assert_equal(res3.unit, units.dimensionless_unscaled)

        # and with arrays inverse!
        res4 = 1 - qf3
        assert_equal(res4.nominal, [-1, -2])
        assert_almost_equal(res4.uncertainty, [0.1, 0.2])
        assert_equal(res4.unit, units.dimensionless_unscaled)

        # array array
        with pytest.raises(UnitsError):
            qf4 - [1, 2]

        with pytest.raises(UnitsError):
            [1, 2] - qf4

        res5 = qf3 - [1, 2]
        assert_equal(res5.nominal, [1, 1])
        assert_almost_equal(res5.uncertainty, [0.1, 0.2])
        assert_equal(res5.unit, units.dimensionless_unscaled)

        res6 = [1, 2] - qf3
        assert_equal(res6.nominal, [-1, -1])
        assert_almost_equal(res6.uncertainty, [0.1, 0.2])
        assert_equal(res6.unit, units.dimensionless_unscaled)

    def test_qfloat_math_sub_inline(self):
        # Sum with another qfloat
        qf = QFloat(1.0, 0.01, 'm')
        i = id(qf)
        qf -= QFloat(0.5, 0.01, 'm')
        assert_equal(qf.nominal, 0.5)
        assert_almost_equal(qf.uncertainty, 0.01414213562373095)
        assert_equal(qf.unit, units.m)
        assert_equal(id(qf), i)

        # raise if incompatible
        with pytest.raises(UnitsError):
            qf -= QFloat(3, unit='s')
        assert_equal(id(qf), i)

        with pytest.raises(UnitsError):
            qf += 1
        assert_equal(id(qf), i)

        # Arrays
        qf = QFloat([1, 2, 3, 4], [0.01, 0.02, 0.03, 0.04], 'cm')
        i = id(qf)
        qf -= QFloat(1, 0.01, 'cm')
        assert_equal(qf.nominal, [0, 1, 2, 3])
        assert_almost_equal(qf.uncertainty, [0.01414213562373095,
                                             0.022360679774997897,
                                             0.03162277660168379,
                                             0.04123105625617661])
        assert_equal(qf.unit, units.cm)
        assert_equal(id(qf), i)

        with pytest.raises(UnitsError):
            qf -= [2, 3, 4, 5]
        assert_equal(id(qf), i)


class Test_QFloat_Mul:
    def test_qfloat_math_mul_single(self):
        qf1 = QFloat(30, 0.5, 's')
        qf2 = QFloat(10, 0.1, 'm')
        qf3 = QFloat(200, 1, 'cm')
        qf4 = QFloat(5, 0.001, None)

        # different dimensions
        res1 = qf1 * qf2
        assert_equal(res1.nominal, 300)
        assert_equal(res1.uncertainty, 5.830951894845301)
        assert_equal(res1.unit, units.s*units.m)
        # inverse gets just the same
        res2 = qf2 * qf1
        assert_equal(res2.nominal, 300)
        assert_equal(res2.uncertainty, 5.830951894845301)
        assert_equal(res2.unit, units.s*units.m)

        # same dimension. Astropy behavior is not convert the units
        res3 = qf2 * qf3
        assert_equal(res3.nominal, 2000)
        assert_equal(res3.uncertainty, 22.360679774997898)
        assert_equal(res3.unit, units.cm*units.m)
        # inverse the same
        res4 = qf2 * qf3
        assert_equal(res4.nominal, 2000)
        assert_equal(res4.uncertainty, 22.360679774997898)
        assert_equal(res4.unit, units.cm*units.m)

        # None (dimensionless)
        res5 = qf1 * qf4
        assert_equal(res5.nominal, 150)
        assert_equal(res5.uncertainty, 2.5001799935204665)
        assert_equal(res5.unit, units.s)
        # inverse the same
        res5 = qf4 * qf1
        assert_equal(res5.nominal, 150)
        assert_equal(res5.uncertainty, 2.5001799935204665)
        assert_equal(res5.unit, units.s)

        # With numbers
        res6 = qf1 * 2
        assert_equal(res6.nominal, 60)
        assert_equal(res6.uncertainty, 1.0)
        assert_equal(res6.unit, units.s)
        # Inverse same
        res6 = 2 * qf1
        assert_equal(res6.nominal, 60)
        assert_equal(res6.uncertainty, 1.0)
        assert_equal(res6.unit, units.s)

        # With units
        res7 = qf1 * units.m
        assert_equal(res7.nominal, 30)
        assert_equal(res7.uncertainty, 0.5)
        assert_equal(res7.unit, units.m*units.s)
        # And with string!
        res9 = qf1 * 'm'
        assert_equal(res9.nominal, 30)
        assert_equal(res9.uncertainty, 0.5)
        assert_equal(res9.unit, units.m*units.s)
        # inverse
        res_a = 'm' * qf1
        assert_equal(res_a.nominal, 30)
        assert_equal(res_a.uncertainty, 0.5)
        assert_equal(res_a.unit, units.m*units.s)

    def test_qfloat_math_mul_array(self):
        qf1 = QFloat(np.arange(5), np.arange(5)*0.01, 'm')
        qf2 = QFloat(20, 0.1, 's')
        qf3 = QFloat(np.arange(16).reshape((4, 4)),
                     np.arange(16).reshape((4, 4))*0.0001, 'km')

        res1 = qf1 * qf1
        assert_equal(res1.nominal, [0, 1, 4, 9, 16])
        assert_almost_equal(res1.uncertainty, [0, 0.01414213562373095, 0.0565685424949238,
                                               0.12727922061357855, 0.2262741699796952])
        assert_equal(res1.unit, units.m*units.m)

        res2 = qf1 * qf2
        assert_equal(res2.nominal, [0, 20, 40, 60, 80])
        assert_almost_equal(res2.uncertainty, [0, 0.223606797749979, 0.447213595499958,
                                               0.6708203932499369, 0.894427190999916])
        assert_equal(res2.unit, units.m*units.s)
        # inverse same
        res3 = qf2 * qf1
        assert_equal(res3.nominal, [0, 20, 40, 60, 80])
        assert_almost_equal(res3.uncertainty, [0, 0.223606797749979, 0.447213595499958,
                                               0.6708203932499369, 0.894427190999916])
        assert_equal(res3.unit, units.m*units.s)

        # with numbers
        res4 = qf1 * 10
        assert_equal(res4.nominal, [0, 10, 20, 30, 40])
        assert_equal(res4.uncertainty, [0, 0.1, 0.2, 0.3, 0.4])
        assert_equal(res4.unit, units.m)
        # same inverse
        res5 = 10 * qf1
        assert_equal(res5.nominal, [0, 10, 20, 30, 40])
        assert_equal(res5.uncertainty, [0, 0.1, 0.2, 0.3, 0.4])
        assert_equal(res5.unit, units.m)

        # 2D Array with qfloat
        res6 = qf2 * qf3
        assert_equal(res6.nominal, [[0, 20, 40, 60],
                                    [80, 100, 120, 140],
                                    [160, 180, 200, 220],
                                    [240, 260, 280, 300]])
        assert_almost_equal(res6.uncertainty, [[0, 0.10002, 0.20004, 0.30005999],
                                               [0.40007999, 0.50009999, 0.60011999, 0.70013999],
                                               [0.80015998, 0.90017998, 1.00019998, 1.10021998],
                                               [1.20023998, 1.30025997, 1.40027997, 1.50029997]])
        assert_equal(res6.unit, units.km*units.s)
        # inverse same
        res7 = qf3 * qf2
        assert_equal(res7.nominal, [[0, 20, 40, 60],
                                    [80, 100, 120, 140],
                                    [160, 180, 200, 220],
                                    [240, 260, 280, 300]])
        assert_almost_equal(res7.uncertainty, [[0, 0.10002, 0.20004, 0.30005999],
                                               [0.40007999, 0.50009999, 0.60011999, 0.70013999],
                                               [0.80015998, 0.90017998, 1.00019998, 1.10021998],
                                               [1.20023998, 1.30025997, 1.40027997, 1.50029997]])
        assert_equal(res7.unit, units.km*units.s)

        # 2D array with numbers
        res8 = 3 * qf3
        assert_equal(res8.nominal,
                     np.arange(16).reshape((4, 4))*3)
        assert_almost_equal(res8.uncertainty,
                            np.arange(16).reshape((4, 4))*0.0001*3)
        assert_equal(res8.unit, units.km)
        # same inverse
        res9 = qf3 * 3
        assert_equal(res9.nominal,
                     np.arange(16).reshape((4, 4))*3)
        assert_almost_equal(res9.uncertainty,
                            np.arange(16).reshape((4, 4))*0.0001*3)
        assert_equal(res9.unit, units.km)

        # With units
        res_a = qf1 * units.m
        assert_equal(res_a.nominal, qf1.nominal)
        assert_equal(res_a.uncertainty, qf1.uncertainty)
        assert_equal(res_a.unit, units.m*units.m)
        # And with string!
        res_c = qf1 * 'm'
        assert_equal(res_c.nominal, qf1.nominal)
        assert_equal(res_c.uncertainty, qf1.uncertainty)
        assert_equal(res_c.unit, units.m*units.m)
        # inverse
        res_b = 'm' * qf1
        assert_equal(res_b.nominal, qf1.nominal)
        assert_equal(res_b.uncertainty, qf1.uncertainty)
        assert_equal(res_b.unit, units.m*units.m)

    def test_qfloat_math_mul_inline(self):
        # single number
        qf1 = QFloat(30, 0.5, 's')
        i = id(qf1)
        qf1 *= QFloat(5, 0.01, None)
        assert_equal(qf1.nominal, 150)
        assert_almost_equal(qf1.uncertainty, 2.5179356624028344)
        assert_equal(qf1.unit, units.s)
        assert_equal(i, id(qf1))
        qf1 *= QFloat(2, 0.1, 'm')
        assert_equal(qf1.nominal, 300)
        assert_almost_equal(qf1.uncertainty, 15.822768405054788)
        assert_equal(qf1.unit, units.s*units.m)
        assert_equal(i, id(qf1))
        qf1 *= 10
        assert_equal(qf1.nominal, 3000)
        assert_almost_equal(qf1.uncertainty, 158.22768405054788)
        assert_equal(qf1.unit, units.s*units.m)
        assert_equal(i, id(qf1))
        qf1 *= 'g'
        assert_equal(qf1.nominal, 3000)
        assert_almost_equal(qf1.uncertainty, 158.22768405054788)
        assert_equal(qf1.unit, units.s*units.m*units.g)
        assert_equal(i, id(qf1))
        qf1 *= units.K
        assert_equal(qf1.nominal, 3000)
        assert_almost_equal(qf1.uncertainty, 158.22768405054788)
        assert_equal(qf1.unit, units.s*units.m*units.g*units.K)
        assert_equal(i, id(qf1))

        # array
        qf2 = QFloat([2, 3, 4], [0.1, 0.2, 0.3], 'm')
        i = id(qf2)
        qf2 *= QFloat(5, 0.01, None)
        assert_almost_equal(qf2.nominal, [10, 15, 20])
        assert_almost_equal(qf2.uncertainty, [0.50039984, 1.0004499, 1.50053324])
        assert_equal(qf2.unit, units.m)
        assert_equal(i, id(qf2))
        qf2 *= QFloat(3.5, 0.1, 'm')
        assert_almost_equal(qf2.nominal, [35, 52.5, 70])
        assert_almost_equal(qf2.uncertainty, [2.01677961, 3.80933393, 5.61979537])
        assert_equal(qf2.unit, units.m*units.m)
        assert_equal(i, id(qf2))
        qf2 *= 10
        assert_almost_equal(qf2.nominal, [350, 525, 700])
        assert_almost_equal(qf2.uncertainty, [20.1677961, 38.0933393, 56.1979537])
        assert_equal(qf2.unit, units.m*units.m)
        assert_equal(i, id(qf2))
        qf2 *= 'g'
        assert_almost_equal(qf2.nominal, [350, 525, 700])
        assert_almost_equal(qf2.uncertainty, [20.1677961, 38.0933393, 56.1979537])
        assert_equal(qf2.unit, units.m*units.m*units.g)
        assert_equal(i, id(qf2))
        qf2 *= units.K
        assert_almost_equal(qf2.nominal, [350, 525, 700])
        assert_almost_equal(qf2.uncertainty, [20.1677961, 38.0933393, 56.1979537])
        assert_equal(qf2.unit, units.m*units.m*units.g*units.K)
        assert_equal(i, id(qf2))


class Test_QFloat_Div:
    def test_qfloat_math_divmod_single(self):
        qf1 = QFloat(20, 0.5, 's')
        qf2 = QFloat(100, 0.1, 'm')
        qf3 = QFloat(2030, 1, 'cm')
        qf4 = QFloat(5, 0.01, None)

        res1 = qf2 / qf1
        assert_equal(res1.nominal, 5)
        assert_almost_equal(res1.uncertainty, 0.12509996003196805)
        assert_equal(res1.unit, units.m/units.s)
        res1f = qf2 // qf1
        assert_equal(res1f.nominal, 5)
        assert_almost_equal(res1f.uncertainty, 0.0)
        assert_equal(res1f.unit, units.m/units.s)
        res1m = qf2 % qf1
        assert_equal(res1m.nominal, 0)
        # assert_almost_equal(res1m.uncertainty, np.inf)
        assert_equal(res1m.unit, units.m)
        # inverse inverse
        res2 = qf1 / qf2
        assert_equal(res2.nominal, 0.2)
        assert_almost_equal(res2.uncertainty, 0.005003998401278721)
        assert_equal(res2.unit, units.s/units.m)
        res2f = qf1 // qf2
        assert_equal(res2f.nominal, 0)
        assert_almost_equal(res2f.uncertainty, 0.0)
        assert_equal(res2f.unit, units.s/units.m)
        res2m = qf1 % qf2
        assert_equal(res2m.nominal, 20)
        assert_almost_equal(res2m.uncertainty, 0.5)
        assert_equal(res2m.unit, units.s)

        # same dimensionality
        res3 = qf3 / qf2
        assert_equal(res3.nominal, 20.3)
        assert_almost_equal(res3.uncertainty, 0.022629405648403586)
        assert_equal(res3.unit, units.cm/units.m)
        res3f = qf3 // qf2
        assert_equal(res3f.nominal, 20)
        assert_almost_equal(res3f.uncertainty, 0.0)
        assert_equal(res3f.unit, units.cm/units.m)
        res3m = qf3 % qf2
        assert_equal(res3m.nominal, 30)
        assert_almost_equal(res3m.uncertainty, 2.23606797749979)
        assert_equal(res3m.unit, units.cm)

        # with no dimensionality
        res4 = qf4 / qf2
        assert_equal(res4.nominal, 0.05)
        assert_almost_equal(res4.uncertainty, 0.00011180339887498949)
        assert_equal(res4.unit, units.Unit('1/m'))
        res4f = qf4 // qf2
        assert_equal(res4f.nominal, 0)
        assert_almost_equal(res4f.uncertainty, 0.0)
        assert_equal(res4f.unit, units.Unit('1/m'))
        res4m = qf4 % qf2
        assert_equal(res4m.nominal, 5)
        assert_almost_equal(res4m.uncertainty, 0.01)
        assert_equal(res4m.unit, units.dimensionless_unscaled)

        # with numbers
        res5 = qf1 / 7
        assert_almost_equal(res5.nominal, 2.857142857142857)
        assert_almost_equal(res5.uncertainty, 0.07142857142857142)
        assert_equal(res5.unit, units.s)
        res5f = qf1 // 7
        assert_equal(res5f.nominal, 2)
        assert_almost_equal(res5f.uncertainty, 0.0)
        assert_equal(res5f.unit, units.s)
        res5m = qf1 % 7
        assert_equal(res5m.nominal, 6)
        assert_almost_equal(res5m.uncertainty, 0.5)
        assert_equal(res5m.unit, units.s)
        # and inverse
        res6 = 70 / qf1
        assert_almost_equal(res6.nominal, 3.5)
        assert_almost_equal(res6.uncertainty, 0.0875)
        assert_equal(res6.unit, units.Unit('1/s'))
        res6f = 70 // qf1
        assert_almost_equal(res6f.nominal, 3.0)
        assert_almost_equal(res6f.uncertainty, 0.0)
        assert_equal(res6f.unit, units.Unit('1/s'))
        res6m = 70 % qf1
        assert_almost_equal(res6m.nominal, 10)
        assert_almost_equal(res6m.uncertainty, 1.5)
        assert_equal(res6m.unit, units.dimensionless_unscaled)

        # with units
        res7 = qf1 / units.m
        assert_equal(res7.nominal, qf1.nominal)
        assert_equal(res7.uncertainty, qf1.uncertainty)
        assert_equal(res7.unit, units.s/units.m)
        res7m = qf1 % units.m
        assert_equal(res7m.unit, units.s)
        # string
        res8 = qf1 / 'm'
        assert_equal(res8.nominal, qf1.nominal)
        assert_equal(res8.uncertainty, qf1.uncertainty)
        assert_equal(res8.unit, units.s/units.m)
        res8m = qf1 % 'm'
        assert_equal(res8m.unit, units.s)

    def test_qfloat_math_divmod_array(self):
        qf1 = QFloat(np.arange(1, 5)*2, np.arange(1, 5)*0.01, 'm')
        qf2 = QFloat(np.arange(1, 5), np.arange(1, 5)*0.01, 's')
        qf3 = QFloat(2, 0.1, 'min')
        qf4 = QFloat(np.arange(1, 17).reshape((4, 4)),
                     np.arange(1, 17).reshape((4, 4))*0.01, 'km')
        qf5 = QFloat(np.arange(1, 17).reshape((4, 4))*4.5,
                     np.arange(1, 17).reshape((4, 4))*0.01, 'h')

        res1 = qf1 / qf2
        assert_equal(res1.nominal, np.ones(4)*2)
        assert_almost_equal(res1.uncertainty, np.ones(4)*0.022360679774997897)
        assert_equal(res1.unit, units.m/units.s)
        res1f = qf1 // qf2
        assert_equal(res1f.nominal, np.ones(4)*2)
        assert_equal(res1f.uncertainty, np.ones(4)*0.0)
        assert_equal(res1f.unit, units.m/units.s)
        res1m = qf1 % qf2
        assert_equal(res1m.nominal, np.ones(4)*0)
        # not continuous
        # assert_equal(res1m.uncertainty, np.ones(4)*0.0)
        assert_equal(res1m.unit, units.m)
        # inverse
        res2 = qf2 / qf1
        assert_equal(res2.nominal, np.ones(4)*0.5)
        assert_almost_equal(res2.uncertainty, np.ones(4)*0.005590169943749474)
        assert_equal(res2.unit, units.s/units.m)
        res2f = qf2 // qf1
        assert_equal(res2f.nominal, np.ones(4)*0)
        assert_equal(res2f.uncertainty, np.ones(4)*0.0)
        assert_equal(res2f.unit, units.s/units.m)
        res2m = qf2 % qf1
        assert_equal(res2m.nominal, [1, 2, 3, 4])
        assert_equal(res2m.uncertainty, [0.01, 0.02, 0.03, 0.04])
        assert_equal(res2m.unit, units.s)

        # 2D arrays
        res3 = qf5 / qf4
        assert_equal(res3.nominal, np.ones(16).reshape((4, 4))*4.5)
        assert_almost_equal(res3.uncertainty,
                            np.ones(16).reshape((4, 4))*0.046097722286464436)
        assert_equal(res3.unit, units.h/units.km)
        res4 = qf5 // qf4
        assert_equal(res4.nominal, np.ones(16).reshape((4, 4))*4.0)
        assert_almost_equal(res4.uncertainty,
                            np.ones(16).reshape((4, 4))*0.0)
        assert_equal(res4.unit, units.h/units.km)
        res4 = qf5 % qf4
        assert_equal(res4.nominal, np.arange(1, 17).reshape((4, 4))*0.5)
        assert_almost_equal(res4.uncertainty,
                            np.arange(1, 17).reshape((4, 4))*0.04123105625617)
        assert_equal(res4.unit, units.h)

        # Array and single
        res5 = qf2 / qf3
        assert_equal(res5.nominal, np.arange(1, 5)*0.5)
        assert_almost_equal(res5.uncertainty,
                            np.arange(1, 5)*0.025495097567963927)
        assert_equal(res5.unit, units.s/units.min)
        res5f = qf2 // qf3
        assert_equal(res5f.nominal, [0, 1, 1, 2])
        assert_almost_equal(res5f.uncertainty, np.zeros(4))
        assert_equal(res5f.unit, units.s/units.min)
        res5m = qf2 % qf3
        assert_equal(res5m.nominal, [1, 0, 1, 0])
        assert_almost_equal(res5m.uncertainty, [0.01, np.inf, 0.1044031, np.inf])
        assert_equal(res5m.unit, units.s)
        # inverse
        res6 = qf3 / qf2
        assert_almost_equal(res6.nominal, [2, 1, 2/3, 0.5])
        assert_almost_equal(res6.uncertainty, [0.10198039027185571,
                                               0.050990195135927854,
                                               0.033993463423951896,
                                               0.025495097567963927])
        assert_equal(res6.unit, units.min/units.s)
        res6f = qf3 // qf2
        assert_equal(res6f.nominal, [2, 1, 0, 0])
        assert_almost_equal(res6f.uncertainty, np.zeros(4))
        assert_equal(res6f.unit, units.min/units.s)
        res6m = qf3 % qf2
        assert_equal(res6m.nominal, [0, 0, 2, 2])
        assert_equal(res6m.unit, units.min)

        # with units
        res7 = qf2 / units.m
        assert_almost_equal(res7.nominal, qf2.nominal)
        assert_almost_equal(res7.uncertainty, qf2.uncertainty)
        assert_equal(res7.unit, units.s/units.m)
        res7m = qf2 % units.m
        assert_equal(res7m.unit, units.s)
        # string
        res8 = qf2 / 'm'
        assert_almost_equal(res8.nominal, qf2.nominal)
        assert_almost_equal(res8.uncertainty, qf2.uncertainty)
        assert_equal(res8.unit, units.s/units.m)
        res8m = qf2 % 'm'
        assert_equal(res8m.unit, units.s)

        # with numbers
        res9 = qf1 / 4
        assert_almost_equal(res9.nominal, np.arange(1, 5)*0.5)
        assert_almost_equal(res9.uncertainty, np.arange(1, 5)*0.0025)
        assert_equal(res9.unit, units.m)
        res9f = qf1 // 4
        assert_almost_equal(res9f.nominal, [0, 1, 1, 2])
        assert_almost_equal(res9f.uncertainty, [0, 0, 0, 0])
        assert_equal(res9f.unit, units.m)
        res9m = qf1 % 4
        assert_almost_equal(res9m.nominal, [2, 0, 2, 0])
        assert_almost_equal(res9m.uncertainty, [0.01, np.nan, 0.03, np.nan])
        assert_equal(res9m.unit, units.m)
        res_a = 8 / qf1
        assert_almost_equal(res_a.nominal, [4, 2, 4/3, 1])
        assert_almost_equal(res_a.uncertainty, [0.02, 0.01, 6/900,
                                                0.005])
        assert_equal(res_a.unit, units.Unit('1/m'))
        res_af = 8 // qf1
        assert_almost_equal(res_af.nominal, [4, 2, 1, 1])
        assert_almost_equal(res_af.uncertainty, [0.0, 0.0, 0.0, 0.0])
        assert_equal(res_af.unit, units.Unit('1/m'))
        res_am = 8 % qf1
        assert_almost_equal(res_am.nominal, [0, 0, 2, 0])
        assert_almost_equal(res_am.std_dev, [np.inf, np.inf, 0.03, np.inf])
        assert_equal(res_am.unit, units.dimensionless_unscaled)

    def test_qfloat_math_truediv_inline(self):
        qf1 = QFloat(30, 0.5, 's')
        qf2 = QFloat(12000, 20, 'm')
        qf3 = QFloat([6000, 3000, 150], [15, 10, 0.5], 'kg')
        qf4 = QFloat([10, 20, 0.1], [0.1, 0.4, 0.01])

        # qf/qf
        i = id(qf2)
        qf2 /= qf1
        assert_equal(qf2.nominal, 400)
        assert_almost_equal(qf2.uncertainty, 6.699917080747261)
        assert_equal(qf2.unit, units.m/units.s)
        assert_equal(i, id(qf2))
        # number
        qf2 /= 2
        assert_equal(qf2.nominal, 200)
        assert_almost_equal(qf2.uncertainty, 3.3499585403736303)
        assert_equal(qf2.unit, units.m/units.s)
        assert_equal(i, id(qf2))
        # unit
        qf2 /= units.s
        assert_equal(qf2.nominal, 200)
        assert_almost_equal(qf2.uncertainty, 3.3499585403736303)
        assert_equal(qf2.unit, units.m/(units.s*units.s))
        assert_equal(i, id(qf2))
        # string
        qf2 /= 'kg'
        assert_equal(qf2.nominal, 200)
        assert_almost_equal(qf2.uncertainty, 3.3499585403736303)
        assert_equal(qf2.unit, units.m/(units.s*units.s*units.kg))
        assert_equal(i, id(qf2))
        # quantity
        qf2 /= 4/units.s
        assert_equal(qf2.nominal, 50)
        assert_almost_equal(qf2.uncertainty, 0.8374896350934076)
        assert_equal(qf2.unit, units.m/(units.s*units.kg))
        assert_equal(i, id(qf2))

        # array / qfloat
        i = id(qf3)
        qf3 /= qf1
        assert_almost_equal(qf3.nominal, [200, 100, 5])
        assert_almost_equal(qf3.uncertainty, [3.370624736026114,
                                              1.699673171197595,
                                              0.08498365855987974])
        assert_equal(qf3.unit, units.kg/units.s)
        assert_equal(i, id(qf3))
        # array / array
        qf3 /= qf4
        assert_almost_equal(qf3.nominal, [20, 5, 50])
        assert_almost_equal(qf3.uncertainty, [0.39193253387682825,
                                              0.13123346456686352,
                                              5.071708018234312])
        assert_equal(qf3.unit, units.kg/units.s)
        assert_equal(i, id(qf3))
        # array / number
        qf3 /= 5
        assert_almost_equal(qf3.nominal, [4, 1, 10])
        assert_almost_equal(qf3.uncertainty, [0.07838650677536566,
                                              0.026246692913372706,
                                              1.0143416036468624])
        assert_equal(qf3.unit, units.kg/units.s)
        assert_equal(i, id(qf3))
        # array  / unit
        qf3 /= units.m
        assert_almost_equal(qf3.nominal, [4, 1, 10])
        assert_almost_equal(qf3.uncertainty, [0.07838650677536566,
                                              0.026246692913372706,
                                              1.0143416036468624])
        assert_equal(qf3.unit, units.kg/(units.s*units.m))
        assert_equal(i, id(qf3))
        # array / string
        qf3 /= 's'
        assert_almost_equal(qf3.nominal, [4, 1, 10])
        assert_almost_equal(qf3.uncertainty, [0.07838650677536566,
                                              0.026246692913372706,
                                              1.0143416036468624])
        assert_equal(qf3.unit, units.kg/(units.s*units.s*units.m))
        assert_equal(i, id(qf3))
        # array / quantity
        qf3 /= 1/units.kg
        assert_almost_equal(qf3.nominal, [4, 1, 10])
        assert_almost_equal(qf3.uncertainty, [0.07838650677536566,
                                              0.026246692913372706,
                                              1.0143416036468624])
        assert_equal(qf3.unit, units.kg*units.kg/(units.s*units.s*units.m))
        assert_equal(i, id(qf3))

    def test_qfloat_math_floordiv_inline(self):
        qf1 = QFloat(30, 0.5, 's')
        qf2 = QFloat(12010, 20, 'm')
        qf3 = QFloat([6040, 3012, 153], [15, 10, 0.5], 'kg')
        qf4 = QFloat([10, 20, 0.1], [0.1, 0.4, 0.01])

        # qf/qf
        i = id(qf2)
        qf2 //= qf1
        assert_equal(qf2.nominal, 400)
        assert_almost_equal(qf2.uncertainty, 0)
        assert_equal(qf2.unit, units.m/units.s)
        assert_equal(i, id(qf2))
        # number
        qf2 //= 7
        assert_equal(qf2.nominal, 57)
        assert_almost_equal(qf2.uncertainty, 0)
        assert_equal(qf2.unit, units.m/units.s)
        assert_equal(i, id(qf2))
        # unit
        qf2 //= units.s
        assert_equal(qf2.nominal, 57)
        assert_almost_equal(qf2.uncertainty, 0)
        assert_equal(qf2.unit, units.m/(units.s*units.s))
        assert_equal(i, id(qf2))
        # string
        qf2 //= 'kg'
        assert_equal(qf2.nominal, 57)
        assert_almost_equal(qf2.uncertainty, 0)
        assert_equal(qf2.unit, units.m/(units.s*units.s*units.kg))
        assert_equal(i, id(qf2))
        # quantity
        qf2 //= 4/units.s
        assert_equal(qf2.nominal, 14)
        assert_almost_equal(qf2.uncertainty, 0.0)
        assert_equal(qf2.unit, units.m/(units.s*units.kg))
        assert_equal(i, id(qf2))

        # array / qfloat
        i = id(qf3)
        qf3 //= qf1
        assert_almost_equal(qf3.nominal, [201, 100, 5])
        assert_almost_equal(qf3.uncertainty, [0, 0, 0])
        assert_equal(qf3.unit, units.kg/units.s)
        assert_equal(i, id(qf3))
        # array / array
        qf3 //= qf4
        assert_almost_equal(qf3.nominal, [20, 5, 49])
        assert_almost_equal(qf3.uncertainty, [0, 0, 0])
        assert_equal(qf3.unit, units.kg/units.s)
        assert_equal(i, id(qf3))
        # array / number
        qf3 //= 5
        assert_almost_equal(qf3.nominal, [4, 1, 9])
        assert_almost_equal(qf3.uncertainty, [0, 0, 0])
        assert_equal(qf3.unit, units.kg/units.s)
        assert_equal(i, id(qf3))
        # array  / unit
        qf3 //= units.m
        assert_almost_equal(qf3.nominal, [4, 1, 9])
        assert_almost_equal(qf3.uncertainty, [0, 0, 0])
        assert_equal(qf3.unit, units.kg/(units.s*units.m))
        assert_equal(i, id(qf3))
        # array / string
        qf3 //= 's'
        assert_almost_equal(qf3.nominal, [4, 1, 9])
        assert_almost_equal(qf3.uncertainty, [0, 0, 0])
        assert_equal(qf3.unit, units.kg/(units.s*units.s*units.m))
        assert_equal(i, id(qf3))
        # array / quantity
        qf3 //= 1/units.kg
        assert_almost_equal(qf3.nominal, [4, 1, 9])
        assert_almost_equal(qf3.uncertainty, [0, 0, 0])
        assert_equal(qf3.unit, units.kg*units.kg/(units.s*units.s*units.m))
        assert_equal(i, id(qf3))

    def test_qfloat_math_mod_inline(self):
        qf1 = QFloat(30, 0.5, 's')
        qf2 = QFloat(12010, 20, 'm')
        qf3 = QFloat([6040, 3012, 153], [15, 10, 0.5], 'kg')
        qf4 = QFloat([7, 5, 2], [0.1, 0.4, 0.01])

        # crazy uncertainties

        # qf/qf
        i = id(qf2)
        qf2 %= qf1
        assert_equal(qf2.nominal, 10)
        assert_equal(qf2.unit, units.m)
        assert_equal(i, id(qf2))
        # number
        qf2 %= 7
        assert_equal(qf2.nominal, 3)
        assert_equal(qf2.unit, units.m)
        assert_equal(i, id(qf2))
        # quantity
        qf2 %= 2/units.s
        assert_equal(qf2.nominal, 1)
        assert_equal(qf2.unit, units.m)
        assert_equal(i, id(qf2))

        # array / qfloat
        i = id(qf3)
        qf3 %= qf1
        assert_almost_equal(qf3.nominal, [10, 12, 3])
        assert_equal(qf3.unit, units.kg)
        assert_equal(i, id(qf3))
        # array / array
        qf3 %= qf4
        assert_almost_equal(qf3.nominal, [3, 2, 1])
        assert_equal(qf3.unit, units.kg)
        assert_equal(i, id(qf3))
        # array / number
        qf3 %= 2
        assert_almost_equal(qf3.nominal, [1, 0, 1])
        assert_equal(qf3.unit, units.kg)
        assert_equal(i, id(qf3))
        # array / quantity
        qf3 %= 1/units.kg
        assert_almost_equal(qf3.nominal, [0, 0, 0])
        assert_equal(qf3.unit, units.kg)
        assert_equal(i, id(qf3))


class Test_QFloat_Pow:
    def test_qfloat_math_power_single(self):
        qf1 = QFloat(2.0, 0.1, 'm')
        qf2 = QFloat(1.5, 0.2, 'cm')
        qf3 = QFloat(0.5, 0.01)
        qf4 = QFloat(4.0, 0.01)

        with pytest.raises(ValueError):
            qf1 ** qf2

        res1 = qf1 ** qf3
        assert_almost_equal(res1.nominal, np.sqrt(2))
        assert_almost_equal(res1.uncertainty, 0.03668910741328604)
        assert_equal(res1.unit, units.m**0.5)

        res2 = qf1 ** qf4
        assert_almost_equal(res2.nominal, 2**4)
        assert_almost_equal(res2.uncertainty, 3.201921235314246)
        assert_equal(res2.unit, units.m*units.m*units.m*units.m)

        res3 = qf1 ** 2
        assert_almost_equal(res3.nominal, 4)
        assert_almost_equal(res3.uncertainty, 0.4)
        assert_equal(res3.unit, units.m*units.m)

        res4 = qf1 ** 0.5
        assert_almost_equal(res4.nominal, np.sqrt(2))
        assert_almost_equal(res4.uncertainty, 0.03535533905932738)
        assert_equal(res4.unit, units.m**0.5)

        # only dimensionless quantity
        with pytest.raises(ValueError):
            qf1 ** (1*units.m)
        res5 = qf1 ** (1*units.dimensionless_unscaled)
        assert_almost_equal(res5.nominal, 2)
        assert_almost_equal(res5.uncertainty, 0.1)
        assert_equal(res5.unit, units.m)

        # string and unit must fail, except for dimensionless
        with pytest.raises(ValueError):
            qf1 ** 's'
        with pytest.raises(ValueError):
            qf1 ** units.s
        res6 = qf1 ** ''
        assert_almost_equal(res6.nominal, 2)
        assert_almost_equal(res6.uncertainty, 0.1)
        assert_equal(res6.unit, units.m)
        res7 = qf1 ** units.dimensionless_unscaled
        assert_almost_equal(res7.nominal, 2)
        assert_almost_equal(res7.uncertainty, 0.1)
        assert_equal(res7.unit, units.m)

        # only dimensionless should power
        with pytest.raises(ValueError):
            1 ** qf1
        res8 = 2 ** qf4
        assert_almost_equal(res8.nominal, 16)
        assert_almost_equal(res8.uncertainty, 0.11090354888959125)
        assert_equal(res8.unit, units.dimensionless_unscaled)

    def test_qfloat_math_power_array(self):
        qf1 = QFloat([2, 3, 4], [0.1, 0.2, 0.3], 'm')
        qf2 = QFloat(1.5, 0.2, 'cm')
        qf3 = QFloat(0.5, 0.01)
        qf4 = QFloat([1, 2, 3], [0.1, 0.2, 0.3])

        with pytest.raises(ValueError):
            qf1 ** qf2

        res1 = qf1 ** qf3
        assert_almost_equal(res1.nominal, np.sqrt([2, 3, 4]))
        assert_almost_equal(res1.uncertainty, [0.03668910741328604,
                                               0.06078995000472617,
                                               0.07996077052073174])
        assert_equal(res1.unit, units.m**0.5)

        with pytest.raises(ValueError):
            qf1 ** qf4

        res3 = qf1 ** 2
        assert_almost_equal(res3.nominal, [4, 9, 16])
        assert_almost_equal(res3.uncertainty, [0.4, 1.2, 2.4])
        assert_equal(res3.unit, units.m*units.m)

        res4 = qf1 ** 0.5
        assert_almost_equal(res4.nominal, np.sqrt([2, 3, 4]))
        assert_almost_equal(res4.uncertainty, [0.03535533905932738,
                                               0.057735026918962574,
                                               0.075])
        assert_equal(res4.unit, units.m**0.5)

        # only dimensionless quantity
        with pytest.raises(ValueError):
            qf1 ** (1*units.m)
        res5 = qf1 ** (1*units.dimensionless_unscaled)
        assert_almost_equal(res5.nominal, [2, 3, 4])
        assert_almost_equal(res5.uncertainty, [0.1, 0.2, 0.3])
        assert_equal(res5.unit, units.m)

        # string and unit must fail, except for dimensionless
        with pytest.raises(ValueError):
            qf1 ** 's'
        with pytest.raises(ValueError):
            qf1 ** units.s
        res6 = qf1 ** ''
        assert_almost_equal(res5.nominal, [2, 3, 4])
        assert_almost_equal(res5.uncertainty, [0.1, 0.2, 0.3])
        assert_equal(res6.unit, units.m)
        res7 = qf1 ** units.dimensionless_unscaled
        assert_almost_equal(res5.nominal, [2, 3, 4])
        assert_almost_equal(res5.uncertainty, [0.1, 0.2, 0.3])
        assert_equal(res7.unit, units.m)

        # only dimensionless size-1 array should work
        with pytest.raises(ValueError):
            1 ** qf1
        with pytest.raises(ValueError):
            1 ** qf2
        with pytest.raises(ValueError):
            1 ** qf4

    def test_qfloat_math_power_inline(self):
        qf1 = QFloat(2.0, 0.1, 'm')
        qf2 = QFloat(1.5, 0.2, 'cm')
        qf3 = QFloat(0.5, 0.01)
        qf4 = QFloat(4.0, 0.01)
        qf5 = QFloat([2, 3, 4], [0.1, 0.2, 0.3], 'm')
        qf6 = QFloat([1, 2, 3], [0.1, 0.2, 0.3])

        i = id(qf1)
        with pytest.raises(ValueError):
            qf1 **= qf2
        with pytest.raises(ValueError):
            qf1 **= qf5
        with pytest.raises(ValueError):
            qf1 **= qf6
        with pytest.raises(ValueError):
            qf1 **= 'm'
        with pytest.raises(ValueError):
            qf1 **= units.s
        assert_equal(i, id(qf1))
        qf1 **= qf3
        assert_almost_equal(qf1.nominal, np.sqrt(2))
        assert_almost_equal(qf1.uncertainty, 0.03668910741328604)
        assert_equal(qf1.unit, units.m**0.5)
        assert_equal(i, id(qf1))
        qf1 **= qf4
        assert_almost_equal(qf1.nominal, 4)
        assert_almost_equal(qf1.uncertainty, 0.4153212953387695)
        assert_equal(qf1.unit, units.m**2)
        assert_equal(i, id(qf1))
        qf1 **= 3
        assert_almost_equal(qf1.nominal, 64)
        assert_almost_equal(qf1.uncertainty, 19.935422176260943)
        assert_equal(qf1.unit, units.m**6)
        assert_equal(i, id(qf1))
        qf1 **= ''
        assert_almost_equal(qf1.nominal, 64)
        assert_almost_equal(qf1.uncertainty, 19.935422176260943)
        assert_equal(qf1.unit, units.m**6)
        assert_equal(i, id(qf1))
        qf1 **= units.dimensionless_unscaled
        assert_almost_equal(qf1.nominal, 64)
        assert_almost_equal(qf1.uncertainty, 19.935422176260943)
        assert_equal(qf1.unit, units.m**6)
        assert_equal(i, id(qf1))

        i = id(qf5)
        with pytest.raises(ValueError):
            qf5 **= qf1
        with pytest.raises(ValueError):
            qf5 **= qf2
        with pytest.raises(ValueError):
            qf5 **= qf6
        with pytest.raises(ValueError):
            qf5 **= 'm'
        with pytest.raises(ValueError):
            qf5 **= units.s
        assert_equal(i, id(qf5))
        qf5 **= qf3
        assert_almost_equal(qf5.nominal, np.sqrt([2, 3, 4]))
        assert_almost_equal(qf5.uncertainty, [0.03668910741328604,
                                              0.06078995000472617,
                                              0.07996077052073174])
        assert_equal(qf5.unit, units.m**0.5)
        assert_equal(i, id(qf5))
        qf5 **= qf4
        assert_almost_equal(qf5.nominal, np.square([2, 3, 4]))
        assert_almost_equal(qf5.uncertainty, [0.4153212953387695,
                                              1.2644622006872943,
                                              2.5611469725808176])
        assert_equal(qf5.unit, units.m**2)
        assert_equal(i, id(qf5))
        qf5 **= 3
        assert_almost_equal(qf5.nominal, [64, 729, 4096])
        assert_almost_equal(qf5.uncertainty, [19.935422176260943,
                                              307.2643147670124,
                                              1966.960874942068])
        assert_equal(qf5.unit, units.m**6)
        assert_equal(i, id(qf5))
        qf5 **= ''
        assert_almost_equal(qf5.nominal, [64, 729, 4096])
        assert_almost_equal(qf5.uncertainty, [19.935422176260943,
                                              307.2643147670124,
                                              1966.960874942068])
        assert_equal(qf5.unit, units.m**6)
        assert_equal(i, id(qf5))
        qf5 **= units.dimensionless_unscaled
        assert_almost_equal(qf5.nominal, [64, 729, 4096])
        assert_almost_equal(qf5.uncertainty, [19.935422176260943,
                                              307.2643147670124,
                                              1966.960874942068])
        assert_equal(qf5.unit, units.m**6)
        assert_equal(i, id(qf5))

    def test_qfloat_pow_special_cases(self):
        qf1 = QFloat(0.0, 0.1, 'm')
        qf2 = qf1**2
        assert_almost_equal(qf2.nominal, 0)
        assert_almost_equal(qf2.uncertainty, 0)
        assert_equal(qf2.unit, 'm^2')

        qf3 = QFloat(1.0, 0.1, 'm')
        qf4 = qf3**1.5
        assert_almost_equal(qf4.nominal, 1)
        assert_almost_equal(qf4.uncertainty, 0.15)
        assert_equal(qf4.unit, 'm(3/2)')

        qf5 = QFloat(1.0, 0.1, 'm')
        qf6 = qf5**0
        assert_almost_equal(qf6.nominal, 1)
        assert_almost_equal(qf6.uncertainty, 0)
        assert_equal(qf6.unit, '')

        qf7 = QFloat(2.0, 0.1, 'm')
        qf8 = qf7**2.5
        assert_almost_equal(qf8.nominal, 2**2.5)
        assert_almost_equal(qf8.uncertainty, 0.1*2.5*2**1.5)
        assert_equal(qf8.unit, 'm(5/2)')


class Test_QFloat_PosNeg:
    def test_qfloat_math_neg(self):
        qf1 = QFloat(1.0, 0.1, 'm')
        qfn1 = -qf1
        assert_equal(qfn1.nominal, -1.0)
        assert_equal(qfn1.uncertainty, 0.1)
        assert_equal(qfn1.unit, units.m)

        qf2 = QFloat(-5, 0.1, 'm')
        qfn2 = -qf2
        assert_equal(qfn2.nominal, 5.0)
        assert_equal(qfn2.uncertainty, 0.1)
        assert_equal(qfn2.unit, units.m)

        qf3 = QFloat([-2, 3, -5, 10], [0.1, 0.4, 0.3, -0.1], 'm')
        qfn3 = -qf3
        assert_equal(qfn3.nominal, [2, -3, 5, -10])
        assert_equal(qfn3.uncertainty, [0.1, 0.4, 0.3, 0.1])
        assert_equal(qfn3.unit, units.m)

    def test_qfloat_math_pos(self):
        qf1 = QFloat(1.0, 0.1, 'm')
        qfn1 = +qf1
        assert_equal(qfn1.nominal, 1.0)
        assert_equal(qfn1.uncertainty, 0.1)
        assert_equal(qfn1.unit, units.m)

        qf2 = QFloat(-5, 0.1, 'm')
        qfn2 = +qf2
        assert_equal(qfn2.nominal, -5.0)
        assert_equal(qfn2.uncertainty, 0.1)
        assert_equal(qfn2.unit, units.m)

        qf3 = QFloat([-2, 3, -5, 10], [0.1, 0.4, 0.3, -0.1], 'm')
        qfn3 = +qf3
        assert_equal(qfn3.nominal, [-2, 3, -5, 10])
        assert_equal(qfn3.uncertainty, [0.1, 0.4, 0.3, 0.1])
        assert_equal(qfn3.unit, units.m)

    def test_qfloat_math_abs(self):
        qf1 = QFloat(1.0, 0.1, 'm')
        qfn1 = abs(qf1)
        assert_equal(qfn1.nominal, 1.0)
        assert_equal(qfn1.uncertainty, 0.1)
        assert_equal(qfn1.unit, units.m)

        qf2 = QFloat(-5, 0.1, 'm')
        qfn2 = abs(qf2)
        assert_equal(qfn2.nominal, 5.0)
        assert_equal(qfn2.uncertainty, 0.1)
        assert_equal(qfn2.unit, units.m)

        qf3 = QFloat([-2, 3, -5, 10], [0.1, 0.4, 0.3, -0.1], 'm')
        qfn3 = abs(qf3)
        assert_equal(qfn3.nominal, [2, 3, 5, 10])
        assert_equal(qfn3.uncertainty, [0.1, 0.4, 0.3, 0.1])
        assert_equal(qfn3.unit, units.m)


class Test_QFloat_Types:
    def test_qfloat_math_float(self):
        qf1 = QFloat(1.5, 0.4, 'm')
        qf2 = QFloat([1.4, 2.5, 3.6], [0.2, 0.3, 0.4], 's')

        assert_equal(float(qf1), 1.5)
        with pytest.raises(TypeError):
            float(qf2)

    def test_qfloat_math_int(self):
        qf1 = QFloat(1.0, 0.4, 'm')
        qf2 = QFloat([1.4, 2.5, 3.6], [0.2, 0.3, 0.4], 's')

        assert_equal(int(qf1), 1)
        with pytest.raises(TypeError):
            int(qf2)
