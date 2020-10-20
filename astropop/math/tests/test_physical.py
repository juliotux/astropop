# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from numpy import testing as npt
import pytest_check as check

from uncertainties import UFloat, unumpy, ufloat

from astropop.math.physical import QFloat, qfloat, unit_property, units, \
                                   same_unit, UnitsError, \
                                   equal_within_errors, ufloat_or_uarray, \
                                   convert_to_qfloat
from astropop.py_utils import check_iterable

# pylint: disable=no-member, pointless-statement


# Test units handling --------------------------------------------------------
def test_qfloat_same_unit():
    qf1 = QFloat(1.0, 0.1, 'm')
    qf2 = QFloat(200, 10, 'cm')
    qf3 = QFloat(120, 0.3, 's')
    qf4 = QFloat(2, 0.01, 'min')

    # must be ok, without any convertion
    qf_1, qf_2 = same_unit(qf1, qf1)
    check.equal(qf_1.nominal, 1.0)
    check.equal(qf_1.uncertainty, 0.1)
    check.equal(qf_1.unit, units.m)
    check.equal(qf_2.nominal, 1.0)
    check.equal(qf_2.uncertainty, 0.1)
    check.equal(qf_2.unit, units.m)

    # Doing conversion
    qf_1, qf_2 = same_unit(qf1, qf2)
    check.equal(qf_1.unit, units.m)
    check.equal(qf_1.nominal, 1.0)
    check.equal(qf_1.uncertainty, 0.1)
    check.equal(qf_2.unit, units.m)
    check.equal(qf_2.nominal, 2.0)
    check.equal(qf_2.uncertainty, 0.1)

    # Inverse Conversion
    qf_2, qf_1 = same_unit(qf2, qf1)
    check.equal(qf_1.unit, units.cm)
    check.equal(qf_1.nominal, 100)
    check.equal(qf_1.uncertainty, 10)
    check.equal(qf_2.unit, units.cm)
    check.equal(qf_2.nominal, 200)
    check.equal(qf_2.uncertainty, 10)

    # Raise conversion
    with pytest.raises(UnitsError):
        same_unit(qf3, qf1, test_qfloat_same_unit)

    # non-prefix conversion
    qf_3, qf_4 = same_unit(qf3, qf4)
    check.equal(qf_3.nominal, 120)
    check.equal(qf_3.uncertainty, 0.3)
    check.equal(qf_3.unit, units.s)
    check.equal(qf_4.nominal, 120)
    check.equal(qf_4.uncertainty, 0.6)
    check.equal(qf_4.unit, units.s)

    # Inverse non-prefix conversion
    qf_4, qf_3 = same_unit(qf4, qf3)
    check.equal(qf_3.nominal, 2)
    check.equal(qf_3.uncertainty, 0.005)
    check.equal(qf_3.unit, units.minute)
    check.equal(qf_4.nominal, 2)
    check.equal(qf_4.uncertainty, 0.01)
    check.equal(qf_4.unit, units.minute)


def test_qfloat_same_unit_array():
    # like for single values, everything must run as arrays
    qf1 = QFloat(1.0, 0.1, 'm')
    qf2 = QFloat(np.arange(0, 2000, 100),
                 np.arange(0, 100, 5),
                 'cm')
    qf3 = QFloat([120, 240],
                 [0.3, 0.6],
                 's')
    qf4 = QFloat([2, 10],
                 [0.01, 0.02],
                 'min')

    # Simplre conversion
    qf_1, qf_2 = same_unit(qf1, qf2, test_qfloat_same_unit_array)
    check.equal(qf_1.nominal, 1.0)
    check.equal(qf_1.uncertainty, 0.1)
    check.equal(qf_1.unit, units.m)
    npt.assert_array_almost_equal(qf_2.nominal, np.arange(0, 20, 1))
    npt.assert_array_almost_equal(qf_2.std_dev, np.arange(0, 1, 0.05))
    check.equal(qf_2.unit, units.m)

    # inverse conversion
    qf_2, qf_1 = same_unit(qf2, qf1, test_qfloat_same_unit_array)
    check.equal(qf_1.nominal, 100)
    check.equal(qf_1.uncertainty, 10)
    check.equal(qf_1.unit, units.cm)
    npt.assert_array_almost_equal(qf_2.nominal, np.arange(0, 2000, 100))
    npt.assert_array_almost_equal(qf_2.std_dev, np.arange(0, 100, 5))
    check.equal(qf_2.unit, units.cm)

    # incompatible
    with pytest.raises(UnitsError):
        same_unit(qf3, qf1, test_qfloat_same_unit)

    qf_3, qf_4 = same_unit(qf3, qf4, test_qfloat_same_unit_array)
    npt.assert_array_almost_equal(qf_3.nominal, [120, 240])
    npt.assert_array_almost_equal(qf_3.std_dev, [0.3, 0.6])
    check.equal(qf_3.unit, units.s)
    npt.assert_array_almost_equal(qf_4.nominal, [120, 600])
    npt.assert_array_almost_equal(qf_4.std_dev, [0.6, 1.2])
    check.equal(qf_4.unit, units.s)


@unit_property
class DummyClass():
    def __init__(self, unit):
        self.unit = unit


@pytest.mark.parametrize('unit,expect', [('meter', units.m),
                                         (units.adu, units.adu),
                                         (None, units.dimensionless_unscaled),
                                         ('', units.dimensionless_unscaled)])
def test_qfloat_unit_property(unit, expect):
    # Getter test
    c = DummyClass(unit)
    check.equal(c.unit, expect)

    # Setter test
    c = DummyClass(None)
    c.unit = unit
    check.equal(c.unit, expect)


def test_qfloat_properties_getset():
    # access all important properties
    qf = QFloat(5.0, 0.025, 'm')
    check.equal(qf.uncertainty, 0.025)
    qf.uncertainty = 0.10
    check.equal(qf.uncertainty, 0.10)
    check.equal(qf.std_dev, 0.10)
    qf.uncertainty = 0.05
    check.equal(qf.std_dev, 0.05)

    # setting nominal resets the uncertainty
    check.equal(qf.nominal, 5.0)
    with pytest.raises(ValueError):
        qf.nominal = None
    check.equal(qf.nominal, 5.0)
    qf.nominal = 10.0
    check.equal(qf.nominal, 10.0)
    check.equal(qf.std_dev, 0.0)
    check.equal(qf.uncertainty, 0.0)

    # with arrays
    qf = QFloat([1, 2, 3], [0.1, 0.2, 0.3], 'm')
    npt.assert_array_almost_equal(qf.uncertainty, [0.1, 0.2, 0.3])
    qf.uncertainty = [0.4, 0.5, 0.6]
    npt.assert_array_almost_equal(qf.uncertainty, [0.4, 0.5, 0.6])
    npt.assert_array_almost_equal(qf.std_dev, [0.4, 0.5, 0.6])
    qf.std_dev = [0.1, 0.2, 0.3]
    npt.assert_array_almost_equal(qf.std_dev, [0.1, 0.2, 0.3])

    npt.assert_array_almost_equal(qf.nominal, [1, 2, 3])
    with pytest.raises(ValueError):
        qf.nominal = None
    npt.assert_array_almost_equal(qf.nominal, [1, 2, 3])
    qf.nominal = [4, 5, 6]
    npt.assert_array_almost_equal(qf.nominal, [4, 5, 6])
    npt.assert_array_almost_equal(qf.std_dev, [0, 0, 0])
    npt.assert_array_almost_equal(qf.uncertainty, [0, 0, 0])


def test_qfloat_properties_reset():
    qf = QFloat(5.0, 0.025, 'm')
    i = id(qf)
    qf.reset(12, 0.2, 's')
    check.equal(i, id(qf))
    check.equal(qf, QFloat(12, 0.2, 's'))

    qf.reset([1, 2, 3], [0.1, 0.2, 0.3], 'm')
    check.equal(i, id(qf))
    npt.assert_array_equal(qf, QFloat([1, 2, 3], [0.1, 0.2, 0.3], 'm'))


def test_qfloat_unit_property_none():
    # Check None and dimensionless_unscaled
    c = DummyClass(None)
    check.is_none(c._unit)
    check.equal(c.unit, units.dimensionless_unscaled)


@pytest.mark.parametrize('unit', ['None', 'Invalid'])
def test_qfloat_unit_property_invalid(unit):
    with pytest.raises(ValueError):
        DummyClass(unit)


def test_qfloat_creation():
    def _create(*args, **kwargs):
        check.is_instance(qfloat(*args, **kwargs), QFloat)

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


def test_qfloat_always_positive_uncertainty():
    qf1 = QFloat(1.0, 0.1, 'm')
    check.equal(qf1.uncertainty, 0.1)

    qf2 = QFloat(1.0, -0.1, 'm')
    check.equal(qf2.uncertainty, 0.1)

    qf3 = QFloat(np.ones(10), np.ones(10)*0.1, 'm')
    npt.assert_array_equal(qf3.uncertainty, np.ones(10)*0.1)

    qf4 = QFloat(np.ones(10), -np.ones(10)*0.1, 'm')
    npt.assert_array_equal(qf4.uncertainty, np.ones(10)*0.1)


def test_qfloat_ufloat_or_uarray_float():
    q = ufloat_or_uarray(QFloat(1.0, 0.1, 'm'))
    check.is_instance(q, UFloat)
    check.equal(q.n, 1.0)
    check.equal(q.s, 0.1)
    check.is_false(check_iterable(q.n))
    check.is_false(check_iterable(q.s))

    # multiple values
    q1, q2 = ufloat_or_uarray(QFloat(1.0, 0.1, 'm'),
                              QFloat(60.0, 0.1, 's'))
    check.is_instance(q1, UFloat)
    check.equal(q1.n, 1.0)
    check.equal(q1.s, 0.1)
    check.is_false(check_iterable(q1.n))
    check.is_false(check_iterable(q1.s))
    check.is_instance(q2, UFloat)
    check.equal(q2.n, 60.0)
    check.equal(q2.s, 0.1)
    check.is_false(check_iterable(q2.n))
    check.is_false(check_iterable(q2.s))


def test_qfloat_ufloat_or_uarray_array():
    a = np.arange(0, 100, 1).reshape((10, 10))
    s = a*0.01
    q = ufloat_or_uarray(QFloat(a, s, 'm'))
    check.is_instance(q, np.ndarray)
    check.is_instance(q[0][0], UFloat)
    n = unumpy.nominal_values(q)
    ns = unumpy.std_devs(q)
    check.equal(q.shape, (10, 10))
    npt.assert_array_equal(n, a)
    npt.assert_array_equal(ns, s)

    # multiple values
    a1 = np.arange(0, 100, 1).reshape((10, 10))
    s1 = a1*0.01
    a2 = [1, 2, 3, 4]
    s2 = [0.1, 0.1, 0.1, 0.1]
    q1, q2 = ufloat_or_uarray(QFloat(a1, s1, 'm'),
                              QFloat(a2, s2, 'm'))
    check.is_instance(q1, np.ndarray)
    check.is_instance(q1[0][0], UFloat)
    n1 = unumpy.nominal_values(q1)
    ns1 = unumpy.std_devs(q1)
    check.equal(q1.shape, (10, 10))
    npt.assert_array_equal(n1, a1)
    npt.assert_array_equal(ns1, s1)
    check.is_instance(q2, np.ndarray)
    check.is_instance(q2[0], UFloat)
    n2 = unumpy.nominal_values(q2)
    ns2 = unumpy.std_devs(q2)
    check.equal(q2.shape, (4,))
    npt.assert_array_equal(n2, a2)
    npt.assert_array_equal(ns2, s2)


@pytest.mark.parametrize('value,expect', [(QFloat(1.0, 0.1, 'm'),
                                           QFloat(1.0, 0.1, 'm')),
                                          (1, QFloat(1.0, 0, None)),
                                          (np.array([1, 2, 3]),
                                           QFloat([1, 2, 3], unit=None)),
                                          ('string', 'raise'),
                                          (None, 'raise'),
                                          (UnitsError, 'raise'),
                                          (1.0*units.m,
                                           QFloat(1.0, 0.0, 'm')),
                                          (unumpy.uarray([0, 0, 0],
                                                         [0, 0, 0]),
                                           QFloat([0, 0, 0], [0, 0, 0],
                                                  None)),
                                          (ufloat(1.0, 0.1),
                                           QFloat(1.0, 0.1, None))
                                          ])
def test_qfloat_converttoqfloat(value, expect):
    if expect == 'raise':
        with pytest.raises(Exception):
            convert_to_qfloat(value)
    else:
        conv = convert_to_qfloat(value)
        npt.assert_array_equal(conv.nominal, expect.nominal)
        npt.assert_array_equal(conv.uncertainty, expect.uncertainty)
        check.equal(conv.unit, expect.unit)


def test_qfloat_unit_conversion():
    qf1 = QFloat(1.0, 0.01, 'm')

    # test converting with string
    qf2 = qf1 << 'cm'
    check.equal(qf2.nominal, 100)
    check.equal(qf2.uncertainty, 1)
    check.equal(qf2.unit, units.cm)

    # test converting using instance
    qf3 = qf1 << units.cm
    check.equal(qf3.nominal, 100)
    check.equal(qf3.uncertainty, 1)
    check.equal(qf3.unit, units.cm)

    # But qf1 must stay the same
    check.equal(qf1.nominal, 1.0)
    check.equal(qf1.uncertainty, 0.01)
    check.equal(qf1.unit, units.m)

    with pytest.raises(Exception):
        # None must fail
        qf1 << None

    with pytest.raises(Exception):
        # incompatible must fail
        qf1 << units.s

    # and this must work with arrays
    qf1 = QFloat([1, 2], [0.1, 0.2], unit='m')
    qf2 = qf1 << 'km'
    npt.assert_array_equal(qf2.nominal, [0.001, 0.002])
    npt.assert_array_equal(qf2.uncertainty, [0.0001, 0.0002])
    check.equal(qf2.unit, units.km)


def test_qfloat_unit_conversion_to():
    qf1 = QFloat(1.0, 0.01, 'm')

    # test converting with string
    qf2 = qf1.to('cm')
    check.equal(qf2.nominal, 100)
    check.equal(qf2.uncertainty, 1)
    check.equal(qf2.unit, units.cm)

    # test converting using instance
    qf3 = qf1.to(units.cm)
    check.equal(qf3.nominal, 100)
    check.equal(qf3.uncertainty, 1)
    check.equal(qf3.unit, units.cm)

    # But qf1 must stay the same
    check.equal(qf1.nominal, 1.0)
    check.equal(qf1.uncertainty, 0.01)
    check.equal(qf1.unit, units.m)

    # inline: now qf1 must change
    i = id(qf1)
    qf1 <<= 'cm'
    check.equal(qf1.nominal, 100)
    check.equal(qf1.uncertainty, 1)
    check.equal(qf1.unit, units.cm)
    check.equal(id(qf1), i)

    with pytest.raises(Exception):
        # None must fail
        qf1.to(None)

    with pytest.raises(Exception):
        # incompatible must fail
        qf1.to(units.s)

    # and this must work with arrays
    qf1 = QFloat([1, 2], [0.1, 0.2], unit='m')
    qf2 = qf1.to('km')
    npt.assert_array_equal(qf2.nominal, [0.001, 0.002])
    npt.assert_array_equal(qf2.uncertainty, [0.0001, 0.0002])
    check.equal(qf2.unit, units.km)


def test_qfloat_getitem():
    # simple array
    qf = QFloat([1, 2, 3, 4, 5],
                [0.1, 0.2, 0.3, 0.4, 0.5],
                's')

    qf1 = qf[0]
    check.equal(qf1.nominal, 1)
    check.equal(qf1.uncertainty, 0.1)
    check.equal(qf1.unit, units.s)

    qf3 = qf[2]
    check.equal(qf3.nominal, 3)
    check.equal(qf3.uncertainty, 0.3)
    check.equal(qf3.unit, units.s)

    qf4 = qf[-1]
    check.equal(qf4.nominal, 5)
    check.equal(qf4.uncertainty, 0.5)
    check.equal(qf4.unit, units.s)

    qf5 = qf[1:4]
    npt.assert_array_equal(qf5.nominal, [2, 3, 4])
    npt.assert_array_equal(qf5.uncertainty, [0.2, 0.3, 0.4])
    check.equal(qf5.unit, units.s)

    with pytest.raises(IndexError):
        qf[10]

    # 2D array
    qf = QFloat(np.arange(1, 17, 1).reshape((4, 4)),
                np.arange(1, 17, 1).reshape((4, 4))*0.01,
                'm')

    qfrow = qf[0]
    npt.assert_array_equal(qfrow.nominal, [1, 2, 3, 4])
    npt.assert_array_equal(qfrow.uncertainty, [0.01, 0.02, 0.03, 0.04])
    check.equal(qfrow.unit, units.m)

    qfcol = qf[:, 1]
    npt.assert_array_equal(qfcol.nominal, [2, 6, 10, 14])
    npt.assert_array_equal(qfcol.uncertainty, [0.02, 0.06, 0.1, 0.14])
    check.equal(qfcol.unit, units.m)

    qf0 = qf[0, 0]
    check.equal(qf0.nominal, 1)
    check.equal(qf0.uncertainty, 0.01)
    check.equal(qf0.unit, units.m)

    qf1 = qf[-1, -1]
    check.equal(qf1.nominal, 16)
    check.equal(qf1.uncertainty, 0.16)
    check.equal(qf1.unit, units.m)

    qfs = qf[2:, 1:3]
    npt.assert_array_equal(qfs.nominal, [[10, 11], [14, 15]])
    npt.assert_array_equal(qfs.uncertainty, [[0.10, 0.11], [0.14, 0.15]])
    check.equal(qfs.unit, units.m)

    with pytest.raises(IndexError):
        qf[10]

    with pytest.raises(IndexError):
        qf[0, 10]

    # Not iterable
    qf = QFloat(10, 0.1, 'm')
    with pytest.raises(TypeError):
        qf[0]


def test_qfloat_setitem():
    # simple array
    qf = QFloat([1, 2, 3, 4, 5],
                [0.1, 0.2, 0.3, 0.4, 0.5],
                's')
    qf[0] = QFloat(10, 0.5, 's')
    qf[-1] = QFloat(-10, unit='min')
    qf[2:4] = QFloat(1, 0.3, 's')
    npt.assert_array_equal(qf.nominal, [10, 2, 1, 1, -600])
    npt.assert_array_equal(qf.uncertainty, [0.5, 0.2, 0.3, 0.3, 0])
    check.equal(qf.unit, units.s)

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
    npt.assert_array_equal(qf.nominal, [[1, 6, 7, 8],
                                        [2, 6, 7, 8],
                                        [3, 10, 20, 20],
                                        [4, 14, 20, 20]])
    npt.assert_almost_equal(qf.uncertainty, [[0.1, 0.4, 0.5, 0.6],
                                             [0.3, 0.06, 0.07, 0.08],
                                             [0.9, 0.1, 0.66, 0.66],
                                             [0.4, 0.14, 0.66, 0.66]])
    check.equal(qf.unit, units.m)

    with pytest.raises(IndexError):
        qf[5, 10] = QFloat(10, 0.1, 'm')
    with pytest.raises(UnitsError):
        qf[0, 0] = 10
    with pytest.raises(UnitsError):
        qf[0, 0] = QFloat(10, 0.1, 's')


def test_qfloat_len():
    with pytest.raises(TypeError):
        len(QFloat(1.0, 0.1, 'm'))
    check.equal(len(QFloat([1], [0.1], 'm')), 1)
    check.equal(len(QFloat([2, 3], [1, 2], 'm')), 2)

    # same behavior of numpy
    check.equal(len(QFloat(np.zeros((10, 5)), np.zeros((10, 5)), 'm')), 10)
    check.equal(len(QFloat(np.zeros((10, 5)), np.zeros((10, 5)), 'm')[0]), 5)

##############################################################################
# Simple math operations -----------------------------------------------------
##############################################################################

# Simple math comparisons -----------------------------------------------------


def test_qfloat_comparison_equality_same_unit():
    # These numbers must not be equal, but equal within errors.
    qf1 = QFloat(1.0, 0.1, 'm')
    qf2 = QFloat(1.05, 0.2, 'm')
    qf3 = QFloat(1.8, 0.1, 'm')
    check.is_false(qf1 == qf2)
    check.is_true(qf1 != qf2)
    check.is_true(qf1 != qf3)
    check.is_true(qf2 != qf3)
    check.is_true(equal_within_errors(qf1, qf2))
    check.is_false(equal_within_errors(qf1, qf3))


def test_qfloat_comparison_equality_convert_units():
    # Units must matter
    qf1 = QFloat(1.0, 0.1, 'm')
    qf3 = QFloat(1.0, 0.1, 'cm')
    qf4 = QFloat(100, 10, 'cm')
    check.is_false(qf1 == qf3)
    check.is_false(qf1 != qf4)
    check.is_true(qf1 != qf3)
    check.is_true(qf1 == qf4)
    check.is_false(equal_within_errors(qf1, qf3))
    check.is_true(equal_within_errors(qf1, qf4))


def test_qfloat_comparison_equality_incompatible_units():
    qf1 = QFloat(1.0, 0.1, 'm')
    qf2 = QFloat(1.0, 0.1, 's')
    check.is_false(qf1 == qf2)
    check.is_true(qf1 != qf2)
    check.is_false(equal_within_errors(qf1, qf2))


def test_qfloat_comparison_inequality_same_unit():
    qf1 = QFloat(1.0, 0.1, 'm')
    qf2 = QFloat(1.0, 0.2, 'm')
    qf3 = QFloat(1.1, 0.1, 'm')
    qf4 = QFloat(1.1, 0.2, 'm')
    qf5 = QFloat(0.9, 0.1, 'm')
    qf6 = QFloat(0.9, 0.5, 'm')

    check.is_true(qf1 <= qf2)
    check.is_false(qf1 < qf2)
    check.is_true(qf1 >= qf2)
    check.is_false(qf1 > qf2)

    check.is_true(qf1 <= qf3)
    check.is_true(qf1 < qf3)
    check.is_false(qf1 > qf3)
    check.is_false(qf1 >= qf3)

    check.is_true(qf1 <= qf4)
    check.is_true(qf1 < qf4)
    check.is_false(qf1 > qf4)
    check.is_false(qf1 >= qf4)

    check.is_true(qf1 >= qf5)
    check.is_true(qf1 > qf5)
    check.is_false(qf1 < qf5)
    check.is_false(qf1 <= qf5)

    check.is_true(qf1 >= qf6)
    check.is_true(qf1 > qf6)
    check.is_false(qf1 < qf6)
    check.is_false(qf1 <= qf6)


def test_qfloat_comparison_inequality_convert_unit():
    qf1 = QFloat(1.0, 0.1, 'm')
    qf2 = QFloat(200, 0.1, 'cm')
    qf3 = QFloat(0.005, 0.0003, 'km')
    qf4 = QFloat(0.0001, 0.00001, 'km')

    check.is_true(qf2 > qf1)
    check.is_true(qf2 >= qf1)
    check.is_false(qf2 < qf1)
    check.is_false(qf2 <= qf1)

    check.is_true(qf3 > qf1)
    check.is_true(qf3 >= qf1)
    check.is_false(qf3 < qf1)
    check.is_false(qf3 <= qf1)

    check.is_true(qf4 < qf1)
    check.is_true(qf4 <= qf1)
    check.is_false(qf4 > qf1)
    check.is_false(qf4 >= qf1)


def test_qfloat_comparison_inequality_incompatible_units():
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


# ADD --------------------------------------------------------

def test_qfloat_math_add_single():
    qf1 = QFloat(1.0, 0.01, 'm')
    qf2 = QFloat(50, 2, 'cm')
    qf3 = QFloat(10, 0.1, None)
    qf4 = QFloat(60, 0.001, 's')

    res1 = qf1 + qf2
    check.equal(res1.nominal, 1.5)
    npt.assert_almost_equal(res1.uncertainty, 0.022360679774)
    check.equal(res1.unit, units.m)

    # same as above, but in cm
    res2 = qf2 + qf1
    check.equal(res2.nominal, 150)
    npt.assert_almost_equal(res2.uncertainty, 2.2360679774)
    check.equal(res2.unit, units.cm)

    # Should fail with incompatible units
    with pytest.raises(UnitsError):
        qf1 + qf3

    with pytest.raises(UnitsError):
        qf1 + qf4


def test_qfloat_math_add_array():
    qf1 = QFloat([1, 2, 3, 4],
                 [0.01, 0.02, 0.01, 0.02],
                 'm')
    qf2 = QFloat([150, 250, 50, 550],
                 [0.1, 5, 0.4, 2],
                 'cm')
    qf3 = QFloat(1, 0.01, 'm')
    qf4 = QFloat(10, 0.1, None)
    qf5 = QFloat(60, 0.001, 's')

    res1 = qf1 + qf2
    npt.assert_array_equal(res1.nominal, [2.5, 4.5, 3.5, 9.5])
    npt.assert_almost_equal(res1.uncertainty, [0.01004987562112089,
                                               0.05385164807134505,
                                               0.010770329614269008,
                                               0.0282842712474619])
    check.equal(res1.unit, units.m)

    # same as above, but in cm
    res2 = qf2 + qf1
    npt.assert_array_equal(res2.nominal, [250, 450, 350, 950])
    npt.assert_almost_equal(res2.uncertainty, [1.004987562112089,
                                               5.385164807134505,
                                               1.0770329614269008,
                                               2.82842712474619])
    check.equal(res2.unit, units.cm)

    # Numpy behavior is to sum arrays with single numbers
    res3 = qf1 + qf3
    npt.assert_array_equal(res3.nominal, [2.0, 3.0, 4.0, 5.0])
    npt.assert_almost_equal(res3.uncertainty, [0.01414213562373095,
                                               0.022360679774997897,
                                               0.01414213562373095,
                                               0.022360679774997897])
    check.equal(res3.unit, units.m)

    # So, it should sum numbers with arrays
    res4 = qf3 + qf1
    npt.assert_array_equal(res4.nominal, [2.0, 3.0, 4.0, 5.0])
    npt.assert_almost_equal(res4.uncertainty, [0.01414213562373095,
                                               0.022360679774997897,
                                               0.01414213562373095,
                                               0.022360679774997897])
    check.equal(res4.unit, units.m)

    # Should fail with incompatible units
    with pytest.raises(UnitsError):
        qf1 + qf4

    with pytest.raises(UnitsError):
        qf1 + qf5


def test_qfloat_math_add_with_numbers():
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
    check.equal(res1.nominal, 3)
    npt.assert_almost_equal(res1.uncertainty, 0.1)
    check.equal(res1.unit, units.dimensionless_unscaled)

    # same as above, but inverse
    res2 = 2 + qf2
    check.equal(res2.nominal, 3)
    npt.assert_almost_equal(res2.uncertainty, 0.1)
    check.equal(res2.unit, units.dimensionless_unscaled)

    # and with arrays!
    res3 = qf3 + 1
    npt.assert_array_equal(res3.nominal, [3, 4])
    npt.assert_almost_equal(res3.uncertainty, [0.1, 0.2])
    check.equal(res3.unit, units.dimensionless_unscaled)

    # and with arrays inverse!
    res4 = 1 + qf3
    npt.assert_array_equal(res4.nominal, [3, 4])
    npt.assert_almost_equal(res4.uncertainty, [0.1, 0.2])
    check.equal(res4.unit, units.dimensionless_unscaled)

    # array array
    with pytest.raises(UnitsError):
        qf4 + [1, 2]

    with pytest.raises(UnitsError):
        [1, 2] + qf4

    res5 = qf3 + [1, 2]
    npt.assert_array_equal(res5.nominal, [3, 5])
    npt.assert_almost_equal(res5.uncertainty, [0.1, 0.2])
    check.equal(res5.unit, units.dimensionless_unscaled)

    res6 = [1, 2] + qf3
    npt.assert_array_equal(res6.nominal, [3, 5])
    npt.assert_almost_equal(res6.uncertainty, [0.1, 0.2])
    check.equal(res6.unit, units.dimensionless_unscaled)


def test_qfloat_math_add_inline():
    # Sum with another qfloat
    qf = QFloat(1.0, 0.1, 'm')
    i = id(qf)
    qf += QFloat(2.0, 0.1, 'm')
    check.equal(qf.nominal, 3)
    npt.assert_almost_equal(qf.uncertainty, 0.1414213562373095)
    check.equal(qf.unit, units.m)
    check.equal(id(qf), i)

    # raise if incompatible
    with pytest.raises(UnitsError):
        qf += QFloat(3, unit='s')
    check.equal(id(qf), i)

    with pytest.raises(UnitsError):
        qf += 1
    check.equal(id(qf), i)

    # Arrays
    qf = QFloat([1, 2, 3, 4],
                [0.01, 0.02, 0.03, 0.04],
                'cm')
    i = id(qf)
    qf += QFloat(1, 0.01, 'cm')
    npt.assert_array_equal(qf.nominal, [2, 3, 4, 5])
    npt.assert_almost_equal(qf.uncertainty, [0.01414213562373095,
                                             0.022360679774997897,
                                             0.03162277660168379,
                                             0.04123105625617661])
    check.equal(qf.unit, units.cm)
    check.equal(id(qf), i)

    with pytest.raises(UnitsError):
        qf += [2, 3, 4, 5]
    check.equal(id(qf), i)


# SUBTRACT -----------------------------------------------------

def test_qfloat_math_sub_single():
    qf1 = QFloat(1.0, 0.01, 'm')
    qf2 = QFloat(50, 2, 'cm')
    qf3 = QFloat(10, 0.1, None)
    qf4 = QFloat(60, 0.001, 's')

    res1 = qf1 - qf2
    check.equal(res1.nominal, 0.5)
    npt.assert_almost_equal(res1.uncertainty, 0.022360679774)
    check.equal(res1.unit, units.m)

    # same as above, but in cm
    res2 = qf2 - qf1
    check.equal(res2.nominal, -50)
    npt.assert_almost_equal(res2.uncertainty, 2.2360679774)
    check.equal(res2.unit, units.cm)

    # Should fail with incompatible units
    with pytest.raises(UnitsError):
        qf1 - qf3

    with pytest.raises(UnitsError):
        qf1 - qf4


def test_qfloat_math_sub_array():
    qf1 = QFloat([1, 2, 3, 4],
                 [0.01, 0.02, 0.01, 0.02],
                 'm')
    qf2 = QFloat([50, 40, 30, 550],
                 [0.1, 5, 0.4, 2],
                 'cm')
    qf3 = QFloat(1, 0.01, 'm')
    qf4 = QFloat(10, 0.1, None)
    qf5 = QFloat(60, 0.001, 's')

    res1 = qf1 - qf2
    npt.assert_array_equal(res1.nominal, [0.5, 1.6, 2.7, -1.5])
    npt.assert_almost_equal(res1.uncertainty, [0.01004987562112089,
                                               0.05385164807134505,
                                               0.010770329614269008,
                                               0.0282842712474619])
    check.equal(res1.unit, units.m)

    # same as above, but in cm
    res2 = qf2 - qf1
    npt.assert_array_equal(res2.nominal, [-50, -160, -270, 150])
    npt.assert_almost_equal(res2.uncertainty, [1.004987562112089,
                                               5.385164807134505,
                                               1.0770329614269008,
                                               2.82842712474619])
    check.equal(res2.unit, units.cm)

    # Numpy behavior is to sum arrays with single numbers
    res3 = qf1 - qf3
    npt.assert_array_equal(res3.nominal, [0.0, 1.0, 2.0, 3.0])
    npt.assert_almost_equal(res3.uncertainty, [0.01414213562373095,
                                               0.022360679774997897,
                                               0.01414213562373095,
                                               0.022360679774997897])
    check.equal(res3.unit, units.m)

    # So, it should sum numbers with arrays
    res4 = qf3 - qf1
    npt.assert_array_equal(res4.nominal, [0.0, -1, -2.0, -3.0])
    npt.assert_almost_equal(res4.uncertainty, [0.01414213562373095,
                                               0.022360679774997897,
                                               0.01414213562373095,
                                               0.022360679774997897])
    check.equal(res4.unit, units.m)

    # Should fail with incompatible units
    with pytest.raises(UnitsError):
        qf1 - qf4

    with pytest.raises(UnitsError):
        qf1 - qf5


def test_qfloat_math_sub_with_numbers():
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
    check.equal(res1.nominal, -1)
    npt.assert_almost_equal(res1.uncertainty, 0.1)
    check.equal(res1.unit, units.dimensionless_unscaled)

    # same as above, but inverse
    res2 = 2 - qf2
    check.equal(res2.nominal, 1)
    npt.assert_almost_equal(res2.uncertainty, 0.1)
    check.equal(res2.unit, units.dimensionless_unscaled)

    # and with arrays!
    res3 = qf3 - 1
    npt.assert_array_equal(res3.nominal, [1, 2])
    npt.assert_almost_equal(res3.uncertainty, [0.1, 0.2])
    check.equal(res3.unit, units.dimensionless_unscaled)

    # and with arrays inverse!
    res4 = 1 - qf3
    npt.assert_array_equal(res4.nominal, [-1, -2])
    npt.assert_almost_equal(res4.uncertainty, [0.1, 0.2])
    check.equal(res4.unit, units.dimensionless_unscaled)

    # array array
    with pytest.raises(UnitsError):
        qf4 - [1, 2]

    with pytest.raises(UnitsError):
        [1, 2] - qf4

    res5 = qf3 - [1, 2]
    npt.assert_array_equal(res5.nominal, [1, 1])
    npt.assert_almost_equal(res5.uncertainty, [0.1, 0.2])
    check.equal(res5.unit, units.dimensionless_unscaled)

    res6 = [1, 2] - qf3
    npt.assert_array_equal(res6.nominal, [-1, -1])
    npt.assert_almost_equal(res6.uncertainty, [0.1, 0.2])
    check.equal(res6.unit, units.dimensionless_unscaled)


def test_qfloat_math_sub_inline():
    # Sum with another qfloat
    qf = QFloat(1.0, 0.01, 'm')
    i = id(qf)
    qf -= QFloat(0.5, 0.01, 'm')
    check.equal(qf.nominal, 0.5)
    npt.assert_almost_equal(qf.uncertainty, 0.01414213562373095)
    check.equal(qf.unit, units.m)
    check.equal(id(qf), i)

    # raise if incompatible
    with pytest.raises(UnitsError):
        qf -= QFloat(3, unit='s')
    check.equal(id(qf), i)

    with pytest.raises(UnitsError):
        qf += 1
    check.equal(id(qf), i)

    # Arrays
    qf = QFloat([1, 2, 3, 4],
                [0.01, 0.02, 0.03, 0.04],
                'cm')
    i = id(qf)
    qf -= QFloat(1, 0.01, 'cm')
    npt.assert_array_equal(qf.nominal, [0, 1, 2, 3])
    npt.assert_almost_equal(qf.uncertainty, [0.01414213562373095,
                                             0.022360679774997897,
                                             0.03162277660168379,
                                             0.04123105625617661])
    check.equal(qf.unit, units.cm)
    check.equal(id(qf), i)

    with pytest.raises(UnitsError):
        qf -= [2, 3, 4, 5]
    check.equal(id(qf), i)


# MULTIPLICATION -------------------------------------------------------------

def test_qfloat_math_mul_single():
    qf1 = QFloat(30, 0.5, 's')
    qf2 = QFloat(10, 0.1, 'm')
    qf3 = QFloat(200, 1, 'cm')
    qf4 = QFloat(5, 0.001, None)

    # different dimensions
    res1 = qf1 * qf2
    check.equal(res1.nominal, 300)
    check.equal(res1.uncertainty, 5.830951894845301)
    check.equal(res1.unit, units.s*units.m)
    # inverse gets just the same
    res2 = qf2 * qf1
    check.equal(res2.nominal, 300)
    check.equal(res2.uncertainty, 5.830951894845301)
    check.equal(res2.unit, units.s*units.m)

    # same dimension. Astropy behavior is not convert the units
    res3 = qf2 * qf3
    check.equal(res3.nominal, 2000)
    check.equal(res3.uncertainty, 22.360679774997898)
    check.equal(res3.unit, units.cm*units.m)
    # inverse the same
    res4 = qf2 * qf3
    check.equal(res4.nominal, 2000)
    check.equal(res4.uncertainty, 22.360679774997898)
    check.equal(res4.unit, units.cm*units.m)

    # None (dimensionless)
    res5 = qf1 * qf4
    check.equal(res5.nominal, 150)
    check.equal(res5.uncertainty, 2.5001799935204665)
    check.equal(res5.unit, units.s)
    # inverse the same
    res5 = qf4 * qf1
    check.equal(res5.nominal, 150)
    check.equal(res5.uncertainty, 2.5001799935204665)
    check.equal(res5.unit, units.s)

    # With numbers
    res6 = qf1 * 2
    check.equal(res6.nominal, 60)
    check.equal(res6.uncertainty, 1.0)
    check.equal(res6.unit, units.s)
    # Inverse same
    res6 = 2 * qf1
    check.equal(res6.nominal, 60)
    check.equal(res6.uncertainty, 1.0)
    check.equal(res6.unit, units.s)

    # With units
    res7 = qf1 * units.m
    check.equal(res7.nominal, 30)
    check.equal(res7.uncertainty, 0.5)
    check.equal(res7.unit, units.m*units.s)
    # TODO: needs numpy ufunc
    # # same inverse
    # res8 = units.m * qf1
    # check.equal(res8.nominal, 30)
    # check.equal(res8.uncertainty, 0.5)
    # check.equal(res8.unit, units.m*units.s)
    # And with string!
    res9 = qf1 * 'm'
    check.equal(res9.nominal, 30)
    check.equal(res9.uncertainty, 0.5)
    check.equal(res9.unit, units.m*units.s)
    # inverse
    resA = 'm' * qf1
    check.equal(resA.nominal, 30)
    check.equal(resA.uncertainty, 0.5)
    check.equal(resA.unit, units.m*units.s)


def test_qfloat_math_mul_array():
    qf1 = QFloat(np.arange(5), np.arange(5)*0.01, 'm')
    qf2 = QFloat(20, 0.1, 's')
    qf3 = QFloat(np.arange(16).reshape((4, 4)),
                 np.arange(16).reshape((4, 4))*0.0001, 'km')

    res1 = qf1 * qf1
    npt.assert_array_equal(res1.nominal, [0, 1, 4, 9, 16])
    npt.assert_almost_equal(res1.uncertainty, [0,
                                               0.01414213562373095,
                                               0.0565685424949238,
                                               0.12727922061357855,
                                               0.2262741699796952])
    check.equal(res1.unit, units.m*units.m)

    res2 = qf1 * qf2
    npt.assert_array_equal(res2.nominal, [0, 20, 40, 60, 80])
    npt.assert_almost_equal(res2.uncertainty, [0,
                                               0.223606797749979,
                                               0.447213595499958,
                                               0.6708203932499369,
                                               0.894427190999916])
    check.equal(res2.unit, units.m*units.s)
    # inverse same
    res3 = qf2 * qf1
    npt.assert_array_equal(res3.nominal, [0, 20, 40, 60, 80])
    npt.assert_almost_equal(res3.uncertainty, [0,
                                               0.223606797749979,
                                               0.447213595499958,
                                               0.6708203932499369,
                                               0.894427190999916])
    check.equal(res3.unit, units.m*units.s)

    # with numbers
    res4 = qf1 * 10
    npt.assert_array_equal(res4.nominal, [0, 10, 20, 30, 40])
    npt.assert_array_equal(res4.uncertainty, [0,
                                              0.1,
                                              0.2,
                                              0.3,
                                              0.4])
    check.equal(res4.unit, units.m)
    # same inverse
    res5 = 10 * qf1
    npt.assert_array_equal(res5.nominal, [0, 10, 20, 30, 40])
    npt.assert_array_equal(res5.uncertainty, [0,
                                              0.1,
                                              0.2,
                                              0.3,
                                              0.4])
    check.equal(res5.unit, units.m)

    # 2D Array with qfloat
    res6 = qf2 * qf3
    npt.assert_array_equal(res6.nominal, [[0, 20, 40, 60],
                                          [80, 100, 120, 140],
                                          [160, 180, 200, 220],
                                          [240, 260, 280, 300]])
    npt.assert_almost_equal(res6.uncertainty, [[0, 0.10002, 0.20004,
                                                0.30005999],
                                               [0.40007999, 0.50009999,
                                                0.60011999, 0.70013999],
                                               [0.80015998, 0.90017998,
                                                1.00019998, 1.10021998],
                                               [1.20023998, 1.30025997,
                                                1.40027997, 1.50029997]])
    check.equal(res6.unit, units.km*units.s)
    # inverse same
    res7 = qf3 * qf2
    npt.assert_array_equal(res7.nominal, [[0, 20, 40, 60],
                                          [80, 100, 120, 140],
                                          [160, 180, 200, 220],
                                          [240, 260, 280, 300]])
    npt.assert_almost_equal(res7.uncertainty, [[0, 0.10002, 0.20004,
                                                0.30005999],
                                               [0.40007999, 0.50009999,
                                                0.60011999, 0.70013999],
                                               [0.80015998, 0.90017998,
                                                1.00019998, 1.10021998],
                                               [1.20023998, 1.30025997,
                                                1.40027997, 1.50029997]])
    check.equal(res7.unit, units.km*units.s)

    # 2D array with numbers
    res8 = 3 * qf3
    npt.assert_array_equal(res8.nominal,
                           np.arange(16).reshape((4, 4))*3)
    npt.assert_almost_equal(res8.uncertainty,
                            np.arange(16).reshape((4, 4))*0.0001*3)
    check.equal(res8.unit, units.km)
    # same inverse
    res9 = qf3 * 3
    npt.assert_array_equal(res9.nominal,
                           np.arange(16).reshape((4, 4))*3)
    npt.assert_almost_equal(res9.uncertainty,
                            np.arange(16).reshape((4, 4))*0.0001*3)
    check.equal(res9.unit, units.km)

    # With units
    resA = qf1 * units.m
    npt.assert_array_equal(resA.nominal, qf1.nominal)
    npt.assert_array_equal(resA.uncertainty, qf1.uncertainty)
    check.equal(resA.unit, units.m*units.m)
    # TODO: needs numpy ufunc
    # # same inverse
    # resB = units.m * qf1
    # npt.assert_array_equal(resB.nominal, qf1.nominal)
    # npt.assert_array_equal(resB.uncertainty, qf1.uncertainty)
    # check.equal(resB.unit, units.m*units.m)
    # And with string!
    resC = qf1 * 'm'
    npt.assert_array_equal(resC.nominal, qf1.nominal)
    npt.assert_array_equal(resC.uncertainty, qf1.uncertainty)
    check.equal(resC.unit, units.m*units.m)
    # inverse
    resB = 'm' * qf1
    npt.assert_array_equal(resB.nominal, qf1.nominal)
    npt.assert_array_equal(resB.uncertainty, qf1.uncertainty)
    check.equal(resB.unit, units.m*units.m)


def test_qfloat_math_mul_inline():
    # single number
    qf1 = QFloat(30, 0.5, 's')
    i = id(qf1)
    qf1 *= QFloat(5, 0.01, None)
    check.equal(qf1.nominal, 150)
    check.equal(qf1.uncertainty, 2.5179356624028344)
    check.equal(qf1.unit, units.s)
    check.equal(i, id(qf1))
    qf1 *= QFloat(2, 0.1, 'm')
    check.equal(qf1.nominal, 300)
    check.equal(qf1.uncertainty, 15.822768405054788)
    check.equal(qf1.unit, units.s*units.m)
    check.equal(i, id(qf1))
    qf1 *= 10
    check.equal(qf1.nominal, 3000)
    check.equal(qf1.uncertainty, 158.22768405054788)
    check.equal(qf1.unit, units.s*units.m)
    check.equal(i, id(qf1))
    qf1 *= 'g'
    check.equal(qf1.nominal, 3000)
    check.equal(qf1.uncertainty, 158.22768405054788)
    check.equal(qf1.unit, units.s*units.m*units.g)
    check.equal(i, id(qf1))
    qf1 *= units.K
    check.equal(qf1.nominal, 3000)
    check.equal(qf1.uncertainty, 158.22768405054788)
    check.equal(qf1.unit, units.s*units.m*units.g*units.K)
    check.equal(i, id(qf1))

    # array
    qf2 = QFloat([2, 3, 4], [0.1, 0.2, 0.3], 'm')
    i = id(qf2)
    qf2 *= QFloat(5, 0.01, None)
    npt.assert_almost_equal(qf2.nominal, [10, 15, 20])
    npt.assert_almost_equal(qf2.uncertainty, [0.50039984, 1.0004499,
                                              1.50053324])
    check.equal(qf2.unit, units.m)
    check.equal(i, id(qf2))
    qf2 *= QFloat(3.5, 0.1, 'm')
    npt.assert_almost_equal(qf2.nominal, [35, 52.5, 70])
    npt.assert_almost_equal(qf2.uncertainty, [2.01677961, 3.80933393,
                                              5.61979537])
    check.equal(qf2.unit, units.m*units.m)
    check.equal(i, id(qf2))
    qf2 *= 10
    npt.assert_almost_equal(qf2.nominal, [350, 525, 700])
    npt.assert_almost_equal(qf2.uncertainty, [20.1677961, 38.0933393,
                                              56.1979537])
    check.equal(qf2.unit, units.m*units.m)
    check.equal(i, id(qf2))
    qf2 *= 'g'
    npt.assert_almost_equal(qf2.nominal, [350, 525, 700])
    npt.assert_almost_equal(qf2.uncertainty, [20.1677961, 38.0933393,
                                              56.1979537])
    check.equal(qf2.unit, units.m*units.m*units.g)
    check.equal(i, id(qf2))
    qf2 *= units.K
    npt.assert_almost_equal(qf2.nominal, [350, 525, 700])
    npt.assert_almost_equal(qf2.uncertainty, [20.1677961, 38.0933393,
                                              56.1979537])
    check.equal(qf2.unit, units.m*units.m*units.g*units.K)
    check.equal(i, id(qf2))

# DIVISION -------------------------------------------------------------------


def test_qfloat_math_divmod_single():
    qf1 = QFloat(20, 0.5, 's')
    qf2 = QFloat(100, 0.1, 'm')
    qf3 = QFloat(2030, 1, 'cm')
    qf4 = QFloat(5, 0.01, None)

    res1 = qf2 / qf1
    check.equal(res1.nominal, 5)
    npt.assert_almost_equal(res1.uncertainty, 0.12509996003196805)
    check.equal(res1.unit, units.m/units.s)
    res1f = qf2 // qf1
    check.equal(res1f.nominal, 5)
    npt.assert_almost_equal(res1f.uncertainty, 0.0)
    check.equal(res1f.unit, units.m/units.s)
    res1m = qf2 % qf1
    check.equal(res1m.nominal, 0)
    # this uncertainty is not continuous
    # npt.assert_almost_equal(res1m.uncertainty, 16777213.75)
    check.equal(res1m.unit, units.m)
    # inverse inverse
    res2 = qf1 / qf2
    check.equal(res2.nominal, 0.2)
    npt.assert_almost_equal(res2.uncertainty, 0.005003998401278721)
    check.equal(res2.unit, units.s/units.m)
    res2f = qf1 // qf2
    check.equal(res2f.nominal, 0)
    npt.assert_almost_equal(res2f.uncertainty, 0.0)
    check.equal(res2f.unit, units.s/units.m)
    res2m = qf1 % qf2
    check.equal(res2m.nominal, 20)
    npt.assert_almost_equal(res2m.uncertainty, 0.5)
    check.equal(res2m.unit, units.s)

    # same dimensionality
    res3 = qf3 / qf2
    check.equal(res3.nominal, 20.3)
    npt.assert_almost_equal(res3.uncertainty, 0.022629405648403586)
    check.equal(res3.unit, units.cm/units.m)
    res3f = qf3 // qf2
    check.equal(res3f.nominal, 20)
    npt.assert_almost_equal(res3f.uncertainty, 0.0)
    check.equal(res3f.unit, units.cm/units.m)
    res3m = qf3 % qf2
    check.equal(res3m.nominal, 30)
    npt.assert_almost_equal(res3m.uncertainty, 2.23606797749979)
    check.equal(res3m.unit, units.cm)

    # with no dimensionality
    res4 = qf4 / qf2
    check.equal(res4.nominal, 0.05)
    npt.assert_almost_equal(res4.uncertainty, 0.00011180339887498949)
    check.equal(res4.unit, 1/units.m)
    res4f = qf4 // qf2
    check.equal(res4f.nominal, 0)
    npt.assert_almost_equal(res4f.uncertainty, 0.0)
    check.equal(res4f.unit, 1/units.m)
    res4m = qf4 % qf2
    check.equal(res4m.nominal, 5)
    npt.assert_almost_equal(res4m.uncertainty, 0.01)
    check.equal(res4m.unit, units.dimensionless_unscaled)

    # with numbers
    res5 = qf1 / 7
    npt.assert_almost_equal(res5.nominal, 2.857142857142857)
    npt.assert_almost_equal(res5.uncertainty, 0.07142857142857142)
    check.equal(res5.unit, units.s)
    res5f = qf1 // 7
    check.equal(res5f.nominal, 2)
    npt.assert_almost_equal(res5f.uncertainty, 0.0)
    check.equal(res5f.unit, units.s)
    res5m = qf1 % 7
    check.equal(res5m.nominal, 6)
    npt.assert_almost_equal(res5m.uncertainty, 0.5)
    check.equal(res5m.unit, units.s)
    # and inverse
    res6 = 70 / qf1
    npt.assert_almost_equal(res6.nominal, 3.5)
    npt.assert_almost_equal(res6.uncertainty, 0.0875)
    check.equal(res6.unit, 1/units.s)
    res6f = 70 // qf1
    npt.assert_almost_equal(res6f.nominal, 3.0)
    npt.assert_almost_equal(res6f.uncertainty, 0.0)
    check.equal(res6f.unit, 1/units.s)
    res6m = 70 % qf1
    npt.assert_almost_equal(res6m.nominal, 10)
    npt.assert_almost_equal(res6m.uncertainty, 1.5)
    check.equal(res6m.unit, units.dimensionless_unscaled)

    # with units
    res7 = qf1 / units.m
    check.equal(res7.nominal, qf1.nominal)
    check.equal(res7.uncertainty, qf1.uncertainty)
    check.equal(res7.unit, units.s/units.m)
    res7m = qf1 % units.m
    check.equal(res7m.unit, units.s)
    # string
    res8 = qf1 / 'm'
    check.equal(res8.nominal, qf1.nominal)
    check.equal(res8.uncertainty, qf1.uncertainty)
    check.equal(res8.unit, units.s/units.m)
    res8m = qf1 % 'm'
    check.equal(res8m.unit, units.s)


def test_qfloat_math_divmod_array():
    qf1 = QFloat(np.arange(1, 5)*2, np.arange(1, 5)*0.01, 'm')
    qf2 = QFloat(np.arange(1, 5), np.arange(1, 5)*0.01, 's')
    qf3 = QFloat(2, 0.1, 'min')
    qf4 = QFloat(np.arange(1, 17).reshape((4, 4)),
                 np.arange(1, 17).reshape((4, 4))*0.01, 'km')
    qf5 = QFloat(np.arange(1, 17).reshape((4, 4))*4.5,
                 np.arange(1, 17).reshape((4, 4))*0.01, 'h')

    res1 = qf1 / qf2
    npt.assert_array_equal(res1.nominal, np.ones(4)*2)
    npt.assert_almost_equal(res1.uncertainty, np.ones(4)*0.022360679774997897)
    check.equal(res1.unit, units.m/units.s)
    res1f = qf1 // qf2
    npt.assert_array_equal(res1f.nominal, np.ones(4)*2)
    npt.assert_array_equal(res1f.uncertainty, np.ones(4)*0.0)
    check.equal(res1f.unit, units.m/units.s)
    res1m = qf1 % qf2
    npt.assert_array_equal(res1m.nominal, np.ones(4)*0)
    # not continuous
    # npt.assert_array_equal(res1m.uncertainty, np.ones(4)*0.0)
    check.equal(res1m.unit, units.m)
    # inverse
    res2 = qf2 / qf1
    npt.assert_array_equal(res2.nominal, np.ones(4)*0.5)
    npt.assert_almost_equal(res2.uncertainty, np.ones(4)*0.005590169943749474)
    check.equal(res2.unit, units.s/units.m)
    res2f = qf2 // qf1
    npt.assert_array_equal(res2f.nominal, np.ones(4)*0)
    npt.assert_array_equal(res2f.uncertainty, np.ones(4)*0.0)
    check.equal(res2f.unit, units.s/units.m)
    res2m = qf2 % qf1
    npt.assert_array_equal(res2m.nominal, [1, 2, 3, 4])
    npt.assert_array_equal(res2m.uncertainty, [0.01, 0.02, 0.03, 0.04])
    check.equal(res2m.unit, units.s)

    # 2D arrays
    res3 = qf5 / qf4
    npt.assert_array_equal(res3.nominal, np.ones(16).reshape((4, 4))*4.5)
    npt.assert_almost_equal(res3.uncertainty,
                            np.ones(16).reshape((4, 4))*0.046097722286464436)
    check.equal(res3.unit, units.h/units.km)
    res4 = qf5 // qf4
    npt.assert_array_equal(res4.nominal, np.ones(16).reshape((4, 4))*4.0)
    npt.assert_almost_equal(res4.uncertainty,
                            np.ones(16).reshape((4, 4))*0.0)
    check.equal(res4.unit, units.h/units.km)
    res4 = qf5 % qf4
    npt.assert_array_equal(res4.nominal, np.arange(1, 17).reshape((4, 4))*0.5)
    npt.assert_almost_equal(res4.uncertainty,
                            np.arange(1, 17).reshape((4, 4))*0.04123105625617)
    check.equal(res4.unit, units.h)

    # Array and single
    res5 = qf2 / qf3
    npt.assert_array_equal(res5.nominal, np.arange(1, 5)*0.5)
    npt.assert_almost_equal(res5.uncertainty,
                            np.arange(1, 5)*0.025495097567963927)
    check.equal(res5.unit, units.s/units.min)
    res5f = qf2 // qf3
    npt.assert_array_equal(res5f.nominal, [0, 1, 1, 2])
    npt.assert_almost_equal(res5f.uncertainty, np.zeros(4))
    check.equal(res5f.unit, units.s/units.min)
    res5m = qf2 % qf3
    npt.assert_array_equal(res5m.nominal, [1, 0, 1, 0])
    # Some are not continuous
    # npt.assert_almost_equal(res5m.uncertainty, np.zeros(4))
    check.equal(res5m.unit, units.s)
    # inverse
    res6 = qf3 / qf2
    npt.assert_almost_equal(res6.nominal, [2, 1, 2/3, 0.5])
    npt.assert_almost_equal(res6.uncertainty, [0.10198039027185571,
                                               0.050990195135927854,
                                               0.033993463423951896,
                                               0.025495097567963927])
    check.equal(res6.unit, units.min/units.s)
    res6f = qf3 // qf2
    npt.assert_array_equal(res6f.nominal, [2, 1, 0, 0])
    npt.assert_almost_equal(res6f.uncertainty, np.zeros(4))
    check.equal(res6f.unit, units.min/units.s)
    res6m = qf3 % qf2
    npt.assert_array_equal(res6m.nominal, [0, 0, 2, 2])
    check.equal(res6m.unit, units.min)

    # with units
    res7 = qf2 / units.m
    npt.assert_almost_equal(res7.nominal, qf2.nominal)
    npt.assert_almost_equal(res7.uncertainty, qf2.uncertainty)
    check.equal(res7.unit, units.s/units.m)
    res7m = qf2 % units.m
    check.equal(res7m.unit, units.s)
    # string
    res8 = qf2 / 'm'
    npt.assert_almost_equal(res8.nominal, qf2.nominal)
    npt.assert_almost_equal(res8.uncertainty, qf2.uncertainty)
    check.equal(res8.unit, units.s/units.m)
    res8m = qf2 % 'm'
    check.equal(res8m.unit, units.s)

    # with numbers
    res9 = qf1 / 4
    npt.assert_almost_equal(res9.nominal, np.arange(1, 5)*0.5)
    npt.assert_almost_equal(res9.uncertainty, np.arange(1, 5)*0.0025)
    check.equal(res9.unit, units.m)
    res9f = qf1 // 4
    npt.assert_almost_equal(res9f.nominal, [0, 1, 1, 2])
    npt.assert_almost_equal(res9f.uncertainty, [0, 0, 0, 0])
    check.equal(res9f.unit, units.m)
    res9m = qf1 % 4
    npt.assert_almost_equal(res9m.nominal, [2, 0, 2, 0])
    npt.assert_almost_equal(res9m.uncertainty, np.arange(1, 5)*0.01)
    check.equal(res9m.unit, units.m)
    resA = 8 / qf1
    npt.assert_almost_equal(resA.nominal, [4, 2, 4/3, 1])
    npt.assert_almost_equal(resA.uncertainty, [0.02, 0.01, 6/900,
                                               0.005])
    check.equal(resA.unit, 1/units.m)
    resAf = 8 // qf1
    npt.assert_almost_equal(resAf.nominal, [4, 2, 1, 1])
    npt.assert_almost_equal(resAf.uncertainty, [0.0, 0.0, 0.0, 0.0])
    check.equal(resAf.unit, 1/units.m)
    resAm = 8 % qf1
    npt.assert_almost_equal(resAm.nominal, [0, 0, 2, 0])
    # npt.assert_almost_equal(resAm.uncertainty, [0.0, 0.0, 0.0, 0.0])
    check.equal(resAm.unit, units.dimensionless_unscaled)


def test_qfloat_math_truediv_inline():
    qf1 = QFloat(30, 0.5, 's')
    qf2 = QFloat(12000, 20, 'm')
    qf3 = QFloat([6000, 3000, 150], [15, 10, 0.5], 'kg')
    qf4 = QFloat([10, 20, 0.1], [0.1, 0.4, 0.01])

    # qf/qf
    i = id(qf2)
    qf2 /= qf1
    check.equal(qf2.nominal, 400)
    npt.assert_almost_equal(qf2.uncertainty, 6.699917080747261)
    check.equal(qf2.unit, units.m/units.s)
    check.equal(i, id(qf2))
    # number
    qf2 /= 2
    check.equal(qf2.nominal, 200)
    npt.assert_almost_equal(qf2.uncertainty, 3.3499585403736303)
    check.equal(qf2.unit, units.m/units.s)
    check.equal(i, id(qf2))
    # unit
    qf2 /= units.s
    check.equal(qf2.nominal, 200)
    npt.assert_almost_equal(qf2.uncertainty, 3.3499585403736303)
    check.equal(qf2.unit, units.m/(units.s*units.s))
    check.equal(i, id(qf2))
    # string
    qf2 /= 'kg'
    check.equal(qf2.nominal, 200)
    npt.assert_almost_equal(qf2.uncertainty, 3.3499585403736303)
    check.equal(qf2.unit, units.m/(units.s*units.s*units.kg))
    check.equal(i, id(qf2))
    # quantity
    qf2 /= 4/units.s
    check.equal(qf2.nominal, 50)
    npt.assert_almost_equal(qf2.uncertainty, 0.8374896350934076)
    check.equal(qf2.unit, units.m/(units.s*units.kg))
    check.equal(i, id(qf2))

    # array / qfloat
    qf3 /= qf1
    npt.assert_array_almost_equal(qf3.nominal, [200, 100, 5])
    npt.assert_array_almost_equal(qf3.uncertainty, [3.370624736026114,
                                                    1.699673171197595,
                                                    0.08498365855987974])
    check.equal(qf3.unit, units.kg/units.s)
    # array / array
    qf3 /= qf4
    npt.assert_array_almost_equal(qf3.nominal, [20, 5, 50])
    npt.assert_array_almost_equal(qf3.uncertainty, [0.39193253387682825,
                                                    0.13123346456686352,
                                                    5.071708018234312])
    check.equal(qf3.unit, units.kg/units.s)
    # array / number
    qf3 /= 5
    npt.assert_array_almost_equal(qf3.nominal, [4, 1, 10])
    npt.assert_array_almost_equal(qf3.uncertainty, [0.07838650677536566,
                                                    0.026246692913372706,
                                                    1.0143416036468624])
    check.equal(qf3.unit, units.kg/units.s)
    # array  / unit
    qf3 /= units.m
    npt.assert_array_almost_equal(qf3.nominal, [4, 1, 10])
    npt.assert_array_almost_equal(qf3.uncertainty, [0.07838650677536566,
                                                    0.026246692913372706,
                                                    1.0143416036468624])
    check.equal(qf3.unit, units.kg/(units.s*units.m))
    # array / string
    qf3 /= 's'
    npt.assert_array_almost_equal(qf3.nominal, [4, 1, 10])
    npt.assert_array_almost_equal(qf3.uncertainty, [0.07838650677536566,
                                                    0.026246692913372706,
                                                    1.0143416036468624])
    check.equal(qf3.unit, units.kg/(units.s*units.s*units.m))
    # array / quantity
    qf3 /= 1/units.kg
    npt.assert_array_almost_equal(qf3.nominal, [4, 1, 10])
    npt.assert_array_almost_equal(qf3.uncertainty, [0.07838650677536566,
                                                    0.026246692913372706,
                                                    1.0143416036468624])
    check.equal(qf3.unit, units.kg*units.kg/(units.s*units.s*units.m))


# POS and NEG ----------------------------------------------------------------


def test_qfloat_math_neg():
    qf1 = QFloat(1.0, 0.1, 'm')
    qfn1 = -qf1
    check.equal(qfn1.nominal, -1.0)
    check.equal(qfn1.uncertainty, 0.1)
    check.equal(qfn1.unit, units.m)

    qf2 = QFloat(-5, 0.1, 'm')
    qfn2 = -qf2
    check.equal(qfn2.nominal, 5.0)
    check.equal(qfn2.uncertainty, 0.1)
    check.equal(qfn2.unit, units.m)

    qf3 = QFloat([-2, 3, -5, 10],
                 [0.1, 0.4, 0.3, -0.1],
                 'm')
    qfn3 = -qf3
    npt.assert_array_equal(qfn3.nominal, [2, -3, 5, -10])
    npt.assert_array_equal(qfn3.uncertainty, [0.1, 0.4, 0.3, 0.1])
    check.equal(qfn3.unit, units.m)


def test_qfloat_math_pos():
    qf1 = QFloat(1.0, 0.1, 'm')
    qfn1 = +qf1
    check.equal(qfn1.nominal, 1.0)
    check.equal(qfn1.uncertainty, 0.1)
    check.equal(qfn1.unit, units.m)

    qf2 = QFloat(-5, 0.1, 'm')
    qfn2 = +qf2
    check.equal(qfn2.nominal, -5.0)
    check.equal(qfn2.uncertainty, 0.1)
    check.equal(qfn2.unit, units.m)

    qf3 = QFloat([-2, 3, -5, 10],
                 [0.1, 0.4, 0.3, -0.1],
                 'm')
    qfn3 = +qf3
    npt.assert_array_equal(qfn3.nominal, [-2, 3, -5, 10])
    npt.assert_array_equal(qfn3.uncertainty, [0.1, 0.4, 0.3, 0.1])
    check.equal(qfn3.unit, units.m)


def test_qfloat_math_abs():
    qf1 = QFloat(1.0, 0.1, 'm')
    qfn1 = abs(qf1)
    check.equal(qfn1.nominal, 1.0)
    check.equal(qfn1.uncertainty, 0.1)
    check.equal(qfn1.unit, units.m)

    qf2 = QFloat(-5, 0.1, 'm')
    qfn2 = abs(qf2)
    check.equal(qfn2.nominal, 5.0)
    check.equal(qfn2.uncertainty, 0.1)
    check.equal(qfn2.unit, units.m)

    qf3 = QFloat([-2, 3, -5, 10],
                 [0.1, 0.4, 0.3, -0.1],
                 'm')
    qfn3 = abs(qf3)
    npt.assert_array_equal(qfn3.nominal, [2, 3, 5, 10])
    npt.assert_array_equal(qfn3.uncertainty, [0.1, 0.4, 0.3, 0.1])
    check.equal(qfn3.unit, units.m)
