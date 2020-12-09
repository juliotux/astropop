# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Math operations with uncertainties and units.

Simplified version of `uncertainties` python package with some
`~astropy.units` addings, in a much more free form.
"""

import numbers
from astropy import units
from astropy.units.quantity_helper.helpers import get_converters_and_unit
from astropy.units.quantity_helper import converters_and_unit, \
                                          check_output
from astropy.units import UnitsError, Quantity
import numpy as np

from ..py_utils import check_iterable
from ._deriv import propagate_2


__all__ = ['unit_property', 'QFloat', 'qfloat', 'units', 'UnitsError',
           'equal_within_errors']


HANDLED_AFUNCS = {}
HANDLED_UFUNCS = {}  # must be func(method, *inputs, **kwargs)


def implements_array_func(numpy_function):
    """Register an __array_function__ implementation for QFloat objects."""
    def decorator_array_func(func):
        HANDLED_AFUNCS[numpy_function] = func
        return func
    return decorator_array_func


def implements_ufunc(numpy_ufunc):
    """Register an ufunc implementation for QFloat objects."""
    def decorator_ufunc(func):
        HANDLED_UFUNCS[numpy_ufunc] = func
        return func
    return decorator_ufunc


def unit_property(cls):
    """Add a `unit` property to a class."""
    def _unit_getter(self):
        if self._unit is None:
            return units.dimensionless_unscaled
        return self._unit

    def _unit_setter(self, value):
        if value is None or units.Unit(value) == units.dimensionless_unscaled:
            self._unit = None
        else:
            self._unit = units.Unit(value)

    cls._unit = None
    cls.unit = property(_unit_getter, _unit_setter,
                        doc="Physical unit of the data.")
    return cls


def convert_to_qfloat(value):
    """Convert a value to QFloat.

    Notes
    -----
    - The `unit` is extracted from a `unit` attribute in the number.
      If this attribute is not present, the number is considered dimensionless.
    - Compilant classes now are:
        * Python standard numbers;
        * Numpy simple arrays.
    """
    # Not change if value is already a qfloat
    if isinstance(value, QFloat):
        return value

    # extract unit (force)
    unit = getattr(value, 'unit', None)

    # number support
    if isinstance(value, numbers.Number):
        # Pure numbers or NDArray don't have uncertainty
        return QFloat(value, None, unit)

    # Numpy arrays support. They need to handle single numbers or ufloat
    if isinstance(value, (np.ndarray, list, tuple)):
        # Astropy Quantities
        if isinstance(value, Quantity):
            return QFloat(value.value, None, value.unit)
        # Everithing else is considered numbers.
        return QFloat(value, None, unit)

    # Handle Astropy units to multuply
    if isinstance(value, (units.UnitBase, str)):
        return QFloat(1.0, 0.0, value)

    # TODO: astropy NDData support?

    raise ValueError(f'Value {value} is not QFloat compilant.')


def require_qfloat(func):
    """Require qfloat as argument decorator."""
    def decorator(self, *others):
        others = [convert_to_qfloat(i) for i in others]
        return func(self, *others)
    return decorator


class _QFloatFormatter():
    """Simple store for numbers that can be rounded to first digit.

    Used mainly for speedup repr and str.
    """
    _rounded = False

    def __init__(self, nominal, std):
        self._n = nominal
        self._s = std
        self._d = np.nan

    @property
    def nominal(self):
        if not self._rounded:
            self.round()
        return self._n

    @property
    def std_dev(self):
        if not self._rounded:
            self.round()
        return self._s

    @property
    def digits(self):
        if not self._rounded:
            self.round()
        return self._d

    def round(self):
        try:
            first_digit = -np.int(np.floor(np.log10(np.abs(self._s))))
            self._n = np.around(self._n, first_digit)
            self._s = np.around(self._s, first_digit)
            self._d = first_digit
        except (ValueError, ZeroDivisionError, OverflowError):
            # Do not change the values
            pass

    def __format__(self, format_spec):
        # For not, format do not matter
        # Positive digits, decimals. Negative digits, integers.
        nominal = self.nominal
        std = self.std_dev
        digits = self.digits

        if not np.isnan(digits):
            if digits > 0:
                n_part = f"{nominal:.{int(digits)}f}"
                s_part = f"{std:.{int(digits)}f}"
            else:
                n_part = f"{nominal:.0f}"
                s_part = f"{std:.0f}"
            return f"{n_part}+-{s_part}"
        else:
            return f"{nominal}+-{std}"

    def __repr__(self):
        return f"{self}"

    def __str__(self):
        return f"{self}"


def create_formater(nominal, std):
    """Create _QFloatFormater handling lists."""
    if np.shape(nominal) != np.shape(std):
        raise ValueError('nominal and std values are incompatilbe.')
    if check_iterable(nominal):
        return [create_formater(n, s) for n, s in zip(nominal, std)]
    return _QFloatFormatter(nominal, std)


def same_unit(qfloat1, qfloat2, func=None):
    """Put 2 qfloats in the same unit."""
    # both units must be the same
    def convert(converter, qf, unit):
        if converter is None:
            return qf
        nom = converter(qf.nominal)
        std = converter(qf.uncertainty)
        return QFloat(nom, std, unit)

    qfloat1, qfloat2 = [convert_to_qfloat(i) for i in (qfloat1, qfloat2)]

    # The error raising require a funcion name
    converters, unit = get_converters_and_unit(func, qfloat1.unit,
                                               qfloat2.unit)
    qfloat1 = convert(converters[0], qfloat1, unit)
    qfloat2 = convert(converters[1], qfloat2, unit)

    return qfloat1, qfloat2


def equal_within_errors(qf1, qf2):
    """Check if two QFloats are equal within errors.

    Parameters
    ----------
    qf1, qf2: `~astropop.math.QFloat`, `float` or `np.ndarray`
        QFloats to compare.

    Returns
    -------
    bool:
        `True` if the numbers are equal within the uncertainties,
        (the difference is smaller then the sum of errors). `False`
        if they are different.

    Notes
    -----
    - We consider two numbers equal within errors when
      number1 - number2 <= error1 + error2
    - Incompatible units means different numbers.
    """
    qf1, qf2 = [convert_to_qfloat(i) for i in (qf1, qf2)]
    try:
        qf1, qf2 = same_unit(qf1, qf2, equal_within_errors)
    except UnitsError:
        # Incompatible units are different numbers.
        return False

    diff = np.abs(qf1.nominal - qf2.nominal)
    erro = qf1.uncertainty + qf2.uncertainty

    return diff <= erro


def qfloat(value, uncertainty=None, unit=None):
    """Create a QFloat from the values.

    Parameters
    ----------
    value: number or array_like
        Nominal value(s) of the quantity.
    uncertainty : number, array_like or `None` (optional)
        Uncertainty value of the quantity. If `None`, the quantity will be
        considered with no errors. Must match `value` shape.
    unit: `~astropy.units.Unit` or string (optional)
        The data unit. Must be `~astropy.units.Unit` compliant.

    Returns
    -------
    f: `~astropop.math.physical.QFloat`
        The QFloat created.
    """
    f = QFloat(value, uncertainty, unit)
    return f


@unit_property
class QFloat():
    """Storing float values with stddev uncertainties and units.

    Parameters
    ----------
    value : number or array_like
        Nominal value(s) of the quantity.
    uncertainty : number, array_like or `None` (optional)
        Uncertainty value of the quantity. If `None`, the quantity will be
        considered with no errors. Must match `value` shape.
    unit : `~astropy.units.Unit` or string (optional)
        The data unit. Must be `~astropy.units.Unit` compliant.

    Notes
    -----
    - This class don't support memmapping. Is intended to be in memory ops.
    - Units are handled by `~astropy.units`.
    - Math operations cares about units and uncertainties.
    """

    _nominal = None
    _uncert = None
    _unit = None

    def __init__(self, value, uncertainty=None, unit=None):
        self.nominal = value
        self.uncertainty = uncertainty
        self.unit = unit

    def _set_uncert(self, value):
        if value is None:
            if check_iterable(self._nominal):
                self._uncert = np.zeros_like(self._nominal)
            else:
                self._uncert = 0.0
        else:
            if np.shape(value) != np.shape(self._nominal):
                raise ValueError('Uncertainty with shape different from '
                                 'nominal value: '
                                 f'{np.shape(value)} '
                                 f'{np.shape(self._nominal)}')
            if check_iterable(self._nominal):
                # Errors must be always positive
                self._uncert = np.abs(np.array(value))
            else:
                self._uncert = float(abs(value))

    def _set_nominal(self, value):
        if value is None:
            raise ValueError('Nominal value cannot be None')
        if check_iterable(value):
            self._nominal = np.array(value)
        else:
            self._nominal = value

        self.uncertainty = None  # always value is reset, uncertainty resets
        # No unit changes

    @property
    def uncertainty(self):
        """Uncertainty of the quantity."""
        return self._uncert

    @uncertainty.setter
    def uncertainty(self, value):
        self._set_uncert(value)

    @property
    def nominal(self):
        """Nominal value of the quantity."""
        return self._nominal

    @nominal.setter
    def nominal(self, value):
        self._set_nominal(value)

    @property
    def std_dev(self):
        """Alias for uncertainty."""
        return self.uncertainty

    @std_dev.setter
    def std_dev(self, value):
        self.uncertainty = value

    @property
    def shape(self):
        return np.shape(self.nominal)

    def reset(self, value, uncertainty=None, unit=None):
        """Reset all the data.

        Parameters
        ----------
        value : number or array_like
            Nominal value(s) of the quantity.
        uncertainty : number, array_like or `None` (optional)
            Uncertainty value of the quantity. If `None`, the quantity will be
            considered with no errors. Must match `value` shape.
        unit : `~astropy.units.Unit` or string (optional)
            The data unit. Must be `~astropy.units.Unit` compliant.
        """
        self.nominal = value
        self.uncertainty = uncertainty
        self.unit = unit

    def to(self, unit):
        """Convert this QFloat to another unit.

        Parameters
        ----------
        - unit: string or `~astropy.units.UnitBase`
            Unit to converto to.

        Returns
        -------
        - QFloat:
            A new instance of this class, converted to the new unit.
        """
        other = units.Unit(unit, parse_strict='silent')
        (_, conv), unit = get_converters_and_unit(self.to, other, self.unit)
        nvalue = conv(self.nominal)  # noqa
        nstd = conv(self.uncertainty)  # noqa
        return QFloat(nvalue, nstd, unit)

    def __repr__(self):
        # FIXME: repr can be very slow for mutch large arrays
        # repr for arrays
        if check_iterable(self.nominal):
            ret = "<QFloat\n"
            ret2 = create_formater(self.nominal, self.std_dev)
            ret2 = np.array(ret2).__repr__()
            ret2 += f'\nunit={str(self.unit)}'
        # repr for single values
        else:
            ret = "<QFloat "
            ret2 = f"{_QFloatFormatter(self.nominal, self.std_dev)}"
            ret2 += f' {str(self.unit)}'
        ret += ret2 + '>'
        return ret

    def __getitem__(self, index):
        """Get one item of given index IF this is iterable."""
        v = self.nominal[index]
        s = self.uncertainty[index]
        return QFloat(v, s, self.unit)

    def __setitem__(self, index, value):
        """Set one item at given index if this is iterable."""
        value = convert_to_qfloat(value)
        _, value = same_unit(self, value, self.__setitem__)

        self._nominal[index] = value.nominal
        self._uncert[index] = value.uncertainty

    def __len__(self):
        return len(self.nominal)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Wrap numpy ufuncs, using uncertainties and units.

        Parameters
        ----------
        function : callable
            Ufunc object that was called.
        method : str
            String indicating which Ufunc method was called
            (``__call__``, ``reduce``, ``reduceat``, ``accumulate``,
            ``outer`` or ``inner``).
        inputs : tuple
            A tuple of the input arguments to the ``ufunc``.
        kwargs : keyword arguments
            A dictionary containing the optional input arguments of the
            ``ufunc``. If given, any out arguments, both positional and
            keyword, are passed as a ``tuple`` in kwargs.

        Returns
        -------
        result : `~astropop.math.QFloat`
            Results of the ufunc, with the unit and uncertainty.

        Notes
        -----
        - Based partially in Astropy's Quantity implementation.
        """
        # Only call supported now
        if method != '__call__':
            return NotImplemented

        if ufunc not in HANDLED_UFUNCS:
            return NotImplemented

        # Get conversion functions to put the inputs in the correct
        # unit
        converters, unit = converters_and_unit(ufunc, method, *inputs)

        # put all inputs as QFloats, a local "require_qfloat"
        inputs = [convert_to_qfloat(i) for i in inputs]

        out = kwargs.get('out', None)
        # Avoid loop back by turning any Quantity output into array views.
        if out is not None:
            # If any out is not a QFloat, must fail
            for o in out:
                if not isinstance(o, QFloat):
                    raise TypeError('All outputs must be QFloats.')

            # If pre-allocated output is used, check it is suitable.
            # This also returns array view, to ensure we don't loop back.
            if ufunc.nout == 1:
                out = out[0]
            # TODO: Check if this is ok
            out_array = check_output(out, unit, inputs, function=ufunc)
            # Ensure output argument remains a tuple.
            kwargs['out'] = (out_array,) if ufunc.nout == 1 else out_array

        arrays = []
        for input_, converter in zip(inputs, converters):
            # This should only work for linear conversions.
            # Is there any non-linear needed?
            nom = converter(input_.nominal)
            std = converter(input_.std_dev)
            arrays.append(unp.uarray(nom, std))

        result = HANDLED_UFUNCS[ufunc](method, *arrays, **kwargs)

        if result is None or result is NotImplemented:
            return result
        # return _return_qfloat_ufunc(result, unit, out)
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        """Wrap numpy functions.

        Parameters
        ----------
        func: callable
            Arbitrary callable exposed by NumPy’s public API.
        types: list
            Collection of unique argument types from the original NumPy
            function call that implement ``__array_function__``.
        args: tuple
            Positional arguments directly passed on from the original call.
        kwargs: dict
            Keyword arguments directly passed on from the original call.
        """
        if func not in HANDLED_AFUNCS:
            return NotImplemented

        return HANDLED_AFUNCS[func](*args, **kwargs)

    def __eq__(self, other):
        try:
            this, other = same_unit(self, other, self.__eq__)
        except Exception:
            # Incompatible units are different numbers.
            # Incompatible types are different
            return False

        if this.nominal != other.nominal or \
           this.uncertainty != other.uncertainty:
            return False
        return True

    def __ne__(self, other):
        return not self == other

    @require_qfloat
    def __gt__(self, other):
        this, other = same_unit(self, other, self.__gt__)
        return this.nominal > other.nominal

    @require_qfloat
    def __ge__(self, other):
        this, other = same_unit(self, other, self.__lt__)
        return this.nominal >= other.nominal

    @require_qfloat
    def __lt__(self, other):
        this, other = same_unit(self, other, self.__lt__)
        return this.nominal < other.nominal

    @require_qfloat
    def __le__(self, other):
        this, other = same_unit(self, other, self.__lt__)
        return this.nominal <= other.nominal

    def __lshift__(self, other):
        """Lshift operator used to convert units."""
        return self.to(other)

    def __ilshift__(self, other):
        res = self << other
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __add__(self, other):
        qf1, qf2 = same_unit(self, other, self.__add__)
        sum_n = qf1.nominal + qf2.nominal
        sum_s = propagate_2('add', sum_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(sum_n, sum_s, qf1.unit)

    @require_qfloat
    def __iadd__(self, other):
        res = self.__add__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __radd__(self, other):
        return other.__add__(self)

    @require_qfloat
    def __sub__(self, other):
        qf1, qf2 = same_unit(self, other, self.__add__)
        sub_n = qf1.nominal - qf2.nominal
        sub_s = propagate_2('add', sub_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(sub_n, sub_s, qf1.unit)

    @require_qfloat
    def __isub__(self, other):
        res = self.__sub__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rsub__(self, other):
        return -self.__sub__(other)

    @require_qfloat
    def __mul__(self, other):
        unit = self.unit * other.unit
        qf1, qf2 = self, other
        mul_n = qf1.nominal * qf2.nominal
        mul_s = propagate_2('mul', mul_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(mul_n, mul_s, unit)

    @require_qfloat
    def __imul__(self, other):
        res = self.__mul__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rmul__(self, other):
        return self.__mul__(other)

    @require_qfloat
    def __truediv__(self, other):
        unit = self.unit / other.unit
        qf1, qf2 = self, other
        div_n = qf1.nominal / qf2.nominal
        div_s = propagate_2('truediv', div_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(div_n, div_s, unit)

    @require_qfloat
    def __itruediv__(self, other):
        res = self.__truediv__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rtruediv__(self, other):
        return other.__truediv__(self)

    @require_qfloat
    def __floordiv__(self, other):
        unit = self.unit / other.unit
        qf1, qf2 = self, other
        div_n = qf1.nominal // qf2.nominal
        div_s = propagate_2('floordiv', div_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(div_n, div_s, unit)

    @require_qfloat
    def __ifloordiv__(self, other):
        res = self.__floordiv__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rfloordiv__(self, other):
        # As the argument always enter here as a qfloat...
        return other.__floordiv__(self)

    @require_qfloat
    def __mod__(self, other):
        qf1, qf2 = self, other
        mod_n = qf1.nominal % qf2.nominal
        mod_s = propagate_2('mod', mod_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(mod_n, mod_s, self.unit)

    @require_qfloat
    def __imod__(self, other):
        res = self.__mod__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rmod__(self, other):
        return other.__mod__(self)

    @require_qfloat
    def __pow__(self, other):
        if other.unit != units.dimensionless_unscaled or \
           not np.isscalar(other.nominal):
            raise ValueError('Power operation size-1 require dimensionless'
                             ' expoent')
        qf1, qf2 = self, other
        pow_n = qf1.nominal ** qf2.nominal
        pow_s = propagate_2('pow', pow_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(pow_n, pow_s, self.unit**other.nominal)

    @require_qfloat
    def __ipow__(self, other):
        res = self.__pow__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rpow__(self, other):
        return other.__pow__(self)

    @require_qfloat
    def __neg__(self):
        return QFloat(-self.nominal, self.uncertainty, self.unit)

    @require_qfloat
    def __pos__(self):
        return QFloat(self.nominal, self.uncertainty, self.unit)

    @require_qfloat
    def __abs__(self):
        return QFloat(np.abs(self.nominal), self.uncertainty, self.unit)

    @require_qfloat
    def __int__(self):
        return np.int(self.nominal)

    @require_qfloat
    def __float__(self):
        return np.float(self.nominal)


# TODO:
# Array functions:
#             - copyto,
#             - atleast_1d, atleast_2d, atleast_3d, broadcast, broadcast_to,
#               expand_dims, squeeze
#             - trunc, ceil
#             - sum, prod, nanprod, nansum, cumprod, cumsum, nancumprod,
#             - nancumsum, diff, ediff1d, cross, square
#             - tile, repeat
#             - concatenate, stack, block, vstack, hstack, dstack, columnstack

@implements_array_func(np.shape)
def qfloat_shape(qf):
    """Implements np.shape for qfloats."""
    return qf.shape


@implements_array_func(np.reshape)
def qfloat_reshape(qf, shape, order='C'):
    """Implements np.reshape for qfloats."""
    nominal = np.reshape(qf.nominal, shape, order)
    std = np.reshape(qf.uncertainty, shape, order)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.ravel)
def qfloat_ravel(qf, order='C'):
    """Implements np.ravel for qfloats."""
    nominal = np.ravel(qf.nominal, order)
    std = np.ravel(qf.uncertainty, order)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.transpose)
def qfloat_transpose(qf, axes=None):
    """Implements np.transpose for qfloats."""
    nominal = np.transpose(qf.nominal, axes)
    std = np.transpose(qf.uncertainty, axes)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.round)
@implements_array_func(np.around)
def qfloat_round(qf, decimals=0, out=None):
    """Implements np.round for qfloats."""
    # out is ignored
    if out is not None:
        raise ValueError('For QFloat, out is ignored.')
    nominal = np.round(qf.nominal, decimals)
    std = np.round(qf.uncertainty, decimals)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.delete)
def qfloat_delete(qf, obj, axis=None):
    """Implements np.delete for qfloats."""
    nominal = np.delete(qf.nominal, obj, axis)
    std = np.delete(qf.uncertainty, obj, axis)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.resize)
def qfloat_resize(qf, new_shape):
    """Implements np.resize for qfloats."""
    nominal = np.resize(qf.nominal, new_shape)
    std = np.resize(qf.uncertainty, new_shape)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.flip)
def qfloat_flip(qf, axis):
    """Implements np.flip for qfloats."""
    # We return a copy and not a view!
    nominal = np.flip(qf.nominal, axis)
    std = np.flip(qf.uncertainty, axis)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.fliplr)
def qfloat_fliplr(qf):
    """Implements np.flip for qfloats."""
    nominal = np.fliplr(qf.nominal)
    std = np.fliplr(qf.uncertainty)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.flipud)
def qfloat_flipud(qf):
    """Implements np.flip for qfloats."""
    nominal = np.flipud(qf.nominal)
    std = np.flipud(qf.uncertainty)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.append)
def qfloat_append(qf, values, axis):
    """Implement np.append for qfloats."""
    # First, convert to the same unit.
    qf1, qf2 = same_unit(qf, values)
    nominal = np.append(qf1.nominal, qf2.nominal, axis)
    std = np.append(qf1.uncertainty, qf2.uncertainty, axis)
    return QFloat(nominal, std, qf1.unit)


@implements_array_func(np.insert)
def qfloat_insert(qf, obj, values, axis):
    """Implement np.insert for qfloats."""
    # First, convert to the same unit.
    qf1, qf2 = same_unit(qf, values)
    nominal = np.insert(qf1.nominal, obj, qf2.nominal, axis)
    std = np.insert(qf1.uncertainty, obj, qf2.uncertainty, axis)
    return QFloat(nominal, std, qf1.unit)


@implements_array_func(np.moveaxis)
def qfloat_moveaxis(qf, source, destination):
    """Implement np.moveaxis for qfloats."""
    nominal = np.moveaxis(qf.nominal, source, destination)
    std = np.moveaxis(qf.uncertainty, source, destination)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.rollaxis)
def qfloat_rollaxis(qf, axis, start=0):
    """Implement np.rollaxis for qfloats."""
    nominal = np.rollaxis(qf.nominal, axis, start)
    std = np.rollaxis(qf.uncertainty, axis, start)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.swapaxes)
def qfloat_rollaxes(qf, axis1, axis2):
    """Implement np.swapaxes for qfloats."""
    nominal = np.swapaxes(qf.nominal, axis1, axis2)
    std = np.swapaxes(qf.uncertainty, axis1, axis2)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.roll)
def qfloat_roll(qf, shift, axis=None):
    """Implement np.roll for qfloats."""
    nominal = np.roll(qf.nominal, shift, axis)
    std = np.roll(qf.uncertainty, shift, axis)
    return QFloat(nominal, std, qf.unit)


@implements_array_func(np.rot90)
def qfloat_rot90(qf, k=1, axes=(0, 1)):
    """Implement np.rot90 for qfloats."""
    nominal = np.rot90(qf.nominal, k, axes)
    std = np.rot90(qf.uncertainty, k, axes)
    return QFloat(nominal, std, qf.unit)


# TODO:
# Numpy ufuncs:
#             - add, subtract, multiply, divide, true_divide, floor_divide,
#               negative, positive, power, float_power, remainder, mod, fmod,
#               divmod, absolute, fabs, rint, sign, exp, exp2, log, log2,
#               log10, expm1, log1p, sqrt, square, cbrt,
#             - sin, cos, tan, arcsin, arccos, arctan, hypot, sinh, cosh,
#               tanh, arcsinh, arccosh, arctanh, degrees, radians, deg2rad,
#               rad2deg
#             - maximum, minimum, fmax, fmin
#             - isfinit, isinf, isnan, fabs, signbit, copysign, modf, fmod,
#               floor, ceil, trunc