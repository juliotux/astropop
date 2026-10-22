# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Math operations with uncertainties and units.

Simplified version of `uncertainties` python package with some
`~astropy.units` addings, in a much more free form.
"""

from __future__ import annotations

import typing as _t
import types as _types
import numpy.typing as _npt
from .._typing import (NumberArray as _NumberArray, QuantityInput as _QuantityInput,
                       UnitLike as _UnitLike, Axis as _Axis, Index as _Index)

import numbers
from functools import partial
from astropy import units
from astropy.units.quantity_helper.helpers import get_converters_and_unit
from astropy.units import UnitsError, Quantity
import numpy as np

from .._unit_property import unit_property
from ._deriv import propagate_2, propagate_1


_F = _t.TypeVar('_F', bound=_t.Callable[..., object])
_R = _t.TypeVar('_R')

__all__ = ['unit_property', 'QFloat', 'qfloat', 'units', 'UnitsError',
           'equal_within_errors', 'UnitsError']


HANDLED_AFUNCS = {}
HANDLED_UFUNCS = {}  # must be func(method, *inputs, **kwargs)


def _implements_array_func(
    numpy_function: _t.Callable[..., object],
) -> _t.Callable[[_F], _F]:
    """Register an __array_function__ implementation for QFloat objects."""
    def decorator_array_func(func: _F) -> _F:
        """Register the callable for the selected NumPy array function and return it."""
        HANDLED_AFUNCS[numpy_function] = func
        return func
    return decorator_array_func


def _implements_ufunc(numpy_ufunc: np.ufunc) -> _t.Callable[[_F], _F]:
    """Register an ufunc implementation for QFloat objects."""
    def decorator_ufunc(func: _F) -> _F:
        """Register the callable for the selected NumPy ufunc and return it."""
        HANDLED_UFUNCS[numpy_ufunc] = func
        return func
    return decorator_ufunc


def convert_to_qfloat(value: _QuantityInput) -> QFloat:
    """Convert a value to QFloat.

    Notes
    -----
    - The ``unit`` is extracted from a ``unit`` attribute in the number.
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

    # FrameData support
    if value.__class__.__name__ == 'FrameData':
        # avoid breaking the code due to cyclic imports
        return QFloat(value.data, value.get_uncertainty(False), value.unit)

    # Handle Astropy units to multuply
    if isinstance(value, (units.UnitBase, str)):
        return QFloat(1.0, 0.0, value)

    # TODO: astropy NDData support?

    raise ValueError(f'Value {value} is not QFloat compilant.')


def require_qfloat(func: _t.Callable[..., _R]) -> _t.Callable[..., _R]:
    """Require qfloat as argument decorator."""
    def decorator(self: QFloat, *others: _QuantityInput) -> _R:
        """Convert the other operands to QFloat before invoking the wrapped method."""
        others = [convert_to_qfloat(i) for i in others]
        return func(self, *others)
    return decorator


def _get_round_digit(std: float, sig: int) -> np.integer:
    """Get the number of digits to round the error."""
    return -np.int_(np.floor(np.log10(np.abs(std)))) + sig - 1


def _round_to_error(
    nom: float,
    std: float,
    sdig: int,
) -> tuple[float, float, float | np.integer]:
    """Round the numbers to the first digit of the error."""
    try:
        dig = _get_round_digit(std, sdig)
        std = np.round(std, dig)
        # repeat the process for handle std>5 in the first digit cases
        dig = _get_round_digit(std, sdig)
        nom = np.round(nom, dig)
        std = np.round(std, dig)
    except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning):
        # Do not change the values
        dig = np.nan
    return nom, std, dig


def _format_qfloat(
    nominal: float,
    std_dev: float,
    sig_digits: int,
    pm_sign: str='+-',
) -> str:
    """Format a qfloat number."""
    # TODO: scientific notation?
    # the numbers need to be rounded for larger than one error.
    nom, std, dig = _round_to_error(nominal, std_dev, sig_digits)
    if not np.isnan(dig):
        if dig > 0:
            return f"{nom:.{int(dig)}f}{pm_sign}{std:.{int(dig)}f}"
        return f"{nom:.0f}{pm_sign}{std:.0f}"
    return f"{nom}{pm_sign}{std}"


class _FormaterElement:
    """Class to format a QFloat array."""

    __slots__ = ('_value', '_std', '_sig_digits')

    def __init__(self, value: float, std: float, sig_digits: int) -> None:
        """Retain one nominal value, its uncertainty and the display precision."""
        self._value = value
        self._std = std
        self._sig_digits = sig_digits

    def __repr__(self) -> str:
        """Format the retained value and uncertainty at the selected precision."""
        return _format_qfloat(self._value, self._std, self._sig_digits)


def _create_formater_array(
    n: _npt.ArrayLike,
    s: _npt.ArrayLike,
    digits: int,
) -> _npt.NDArray:
    """Create an array of _FormaterElements."""
    if np.isscalar(n):
        return _FormaterElement(n, s, digits)
    else:
        return [_create_formater_array(ni, si, digits) for ni, si in zip(n, s)]


def same_unit(
    qfloat1: _QuantityInput,
    qfloat2: _QuantityInput,
    func: _t.Callable[..., object] | None=None,
) -> tuple[QFloat, QFloat]:
    """Put 2 qfloats in the same unit."""
    # both units must be the same
    def convert(
        converter: _t.Callable[[object], object] | None,
        qf: QFloat,
        unit: _UnitLike,
    ) -> QFloat:
        """Apply a unit converter to nominal values and uncertainties when needed."""
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


def equal_within_errors(
    qf1: _QuantityInput,
    qf2: _QuantityInput,
) -> bool | _npt.NDArray[np.bool_]:
    """Check if two QFloats are equal within errors.

    Parameters
    ----------
    qf1, qf2
        QFloats to compare.

    Returns
    -------
    bool
        `True` if the numbers are equal within the uncertainties,
        (the difference is smaller then the sum of errors). `False`
        if they are different.

    Notes
    -----
    - We consider two numbers equal within errors when
      abs(number1 - number2) <= error1 + error2
    - Incompatible units means different numbers.

    Examples
    --------
    >>> from astropop.math import QFloat
    >>> from astropop.math.physical import equal_within_errors
    >>> first = QFloat(1., 0.1, 'm')
    >>> second = QFloat(115., 10., 'cm')

    >>> # Compare the difference with the sum of errors in compatible units.
    >>> print(bool(equal_within_errors(first, second)))
    True
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


def qfloat(
    value: _QuantityInput,
    uncertainty: _npt.ArrayLike | None=None,
    unit: _UnitLike=None,
) -> QFloat:
    """Create a QFloat from the values.

    Parameters
    ----------
    value
        Nominal value(s) of the quantity.
    uncertainty
        Uncertainty value of the quantity. If `None`, the quantity will be
        considered with no errors. Must match ``value`` shape.
    unit
        The data unit. Must be `~astropy.units.Unit` compliant.

    Returns
    -------
    f
        The QFloat created.

    Examples
    --------
    >>> from astropop.math import qfloat
    >>> measurement = qfloat(10., 0.5, 'adu')
    >>> print(measurement.nominal, measurement.uncertainty, measurement.unit)
    10.0 0.5 adu
    """
    f = QFloat(value, uncertainty, unit)
    return f


@unit_property
class QFloat():
    """Storing float values with stddev uncertainties and units.

    Parameters
    ----------
    value
        Nominal value(s) of the quantity. Must be a real number, array of real
        numbers or a QFloat.
    uncertainty
        Uncertainty value of the quantity. If `None`, the quantity will be
        considered with no errors. Must match ``value`` shape.
    unit
        The data unit. Must be `~astropy.units.Unit` compliant.

    Notes
    -----
    - This class don't support memmapping. Is intended to be in memory ops.
    - Units are handled by `~astropy.units`.
    - Math operations cares about units and uncertainties.

    Examples
    --------
    >>> from astropop.math import QFloat
    >>> length = QFloat(2., 0.1, 'm')

    >>> # Multiplying by an exact scalar scales the standard uncertainty too.
    >>> doubled = length * 3
    >>> print(f'{doubled.nominal:.1f} +/- {doubled.uncertainty:.1f} {doubled.unit}')
    6.0 +/- 0.3 m

    >>> # Arrays carry one standard uncertainty per nominal value.
    >>> samples = QFloat([1., 2.], [0.1, 0.2], 'm')
    >>> print(samples.shape, samples.uncertainty.tolist())
    (2,) [0.1, 0.2]
    """

    # The decorator installs this descriptor at runtime; expose both access
    # contracts to static tools without replacing the runtime implementation.
    if _t.TYPE_CHECKING:
        @property
        def unit(self) -> units.UnitBase:
            """Return the physical unit, using dimensionless units when none was stored."""
            ...

        @unit.setter
        def unit(self, value: _UnitLike) -> None:
            """Normalize and replace the unit without converting nominal values or errors."""
            ...

    _nominal = None
    _uncert = None
    _unit = None
    _sig_digits = 1

    def __init__(
        self,
        value: _QuantityInput,
        uncertainty: _npt.ArrayLike | None=None,
        unit: _UnitLike=None,
    ) -> None:
        """Normalize nominal values, initialize standard uncertainties and assign the unit."""
        value, uncertainty, unit = self._check_inputs(value, uncertainty, unit)
        self._nominal = value
        self._set_uncert(uncertainty)
        self.unit = unit

    def _check_inputs(
        self,
        value: _QuantityInput,
        uncertainty: _npt.ArrayLike | None=None,
        unit: _UnitLike=None,
    ) -> tuple[_NumberArray, _NumberArray | None, _UnitLike]:
        """Normalize real nominal values and extract an existing QFloat's components.

        Explicit uncertainty or unit overrides are rejected for QFloat inputs.
        Nonscalar nominal values are copied to floating-point arrays.
        """
        if isinstance(value, QFloat):
            qf = value
            value = value.nominal
            if uncertainty is not None:
                raise ValueError('uncertainty must be None if value is a '
                                 'QFloat.')
            uncertainty = qf.uncertainty
            if unit is not None:
                raise ValueError('unit must be None if value is a QFloat.')
            unit = qf.unit
        if np.any(np.array(value) == None):  # noqa: E711
            raise TypeError('value must be not None.')
        for i in value, uncertainty:
            if not np.any(np.isreal(i)):
                raise TypeError('value and uncertainty must be real numbers, '
                                'or arrays of real numbers.')
        if not np.isscalar(value):
            value = np.array(value, dtype=float)
        else:
            value = float(value)
        return value, uncertainty, unit

    def _set_uncert(self, value: _npt.ArrayLike | None) -> None:
        """Store absolute uncertainties matching the nominal shape.

        None creates zero uncertainties. None entries in an uncertainty array
        are replaced by zero; mismatched shapes raise ValueError.
        """
        if value is None:
            if not np.isscalar(self._nominal):
                self._uncert = np.zeros_like(self._nominal)
            else:
                self._uncert = 0.0
        else:
            if not np.any(np.isreal(value)):
                raise TypeError('uncertainty must be real numbers')
            if np.shape(value) != np.shape(self._nominal):
                raise ValueError('Uncertainty with shape different from '
                                 'nominal value: '
                                 f'{np.shape(value)} '
                                 f'{np.shape(self._nominal)}')
            if not np.isscalar(self._nominal):
                # Errors must be always positive
                value = np.array(value)
                value[value == None] = 0.0  # noqa: E711
                self._uncert = np.abs(np.array(value))
            else:
                self._uncert = float(abs(value))

    def _set_nominal(self, value: _npt.ArrayLike) -> None:
        """Replace nominal values and reset their uncertainties.

        A tuple can supply value and uncertainty, optionally followed by a unit.
        Without a supplied unit the current unit is retained.
        """
        uncertainty = None
        unit = None
        if isinstance(value, tuple):
            if len(value) == 2:
                value, uncertainty = value
            else:
                value, uncertainty, unit = value
        value, uncertainty, unit = self._check_inputs(value, uncertainty, unit)
        self._nominal = value
        self._set_uncert(uncertainty)
        if unit is not None:
            self.unit = unit

    @property
    def uncertainty(self) -> _NumberArray:
        """Uncertainty of the quantity."""
        return self._uncert

    @uncertainty.setter
    def uncertainty(self, value: _npt.ArrayLike | None) -> None:
        """Replace standard uncertainties after checking shape and taking absolute values."""
        self._set_uncert(value)

    @property
    def nominal(self) -> _NumberArray:
        """Nominal value of the quantity."""
        return self._nominal

    @nominal.setter
    def nominal(self, value: _npt.ArrayLike) -> None:
        """Replace nominal values, accepting optional uncertainty and unit tuple entries."""
        self._set_nominal(value)

    @property
    def std_dev(self) -> _NumberArray:
        """Alias for uncertainty."""
        return self.uncertainty

    @std_dev.setter
    def std_dev(self, value: _npt.ArrayLike | None) -> None:
        """Assign standard uncertainties through the uncertainty property."""
        self.uncertainty = value

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the quantity."""
        return np.shape(self.nominal)

    @property
    def size(self) -> int:
        """Number of elements in the quantity."""
        return np.size(self.nominal)

    @property
    def sig_digits(self) -> int:
        """Number of significant digits."""
        return self._sig_digits

    @sig_digits.setter
    def sig_digits(self, value: int) -> None:
        """Set display precision by converting a real value to an integer."""
        if not np.isreal(value):
            raise TypeError('sig_digits must be a real number.')
        self._sig_digits = int(value)

    def reset(
        self,
        value: _QuantityInput,
        uncertainty: _npt.ArrayLike | None=None,
        unit: _UnitLike=None,
    ) -> None:
        """Reset all the data.

        Parameters
        ----------
        value
            Nominal value(s) of the quantity.
        uncertainty
            Uncertainty value of the quantity. If `None`, the quantity will be
            considered with no errors. Must match ``value`` shape.
        unit
            The data unit. Must be `~astropy.units.Unit` compliant.
        """
        self.nominal = value
        self.uncertainty = uncertainty
        self.unit = unit

    def to(self, unit: _UnitLike) -> QFloat:
        """Convert this QFloat to another unit.

        Parameters
        ----------
        - unit: string or `~astropy.units.UnitBase`
            Unit to converto to.

        Returns
        -------
        result
            A new instance of this class, converted to the new unit.

        Examples
        --------
        >>> from astropop.math import QFloat
        >>> meters = QFloat(1., 0.02, 'm')
        >>> centimeters = meters.to('cm')

        >>> # Convert both the nominal value and its standard uncertainty.
        >>> print(centimeters.nominal, centimeters.uncertainty, centimeters.unit)
        100.0 2.0 cm
        >>> print(meters.nominal, meters.unit)
        1.0 m
        """
        other = units.Unit(unit, parse_strict='silent')
        (_, conv), unit = get_converters_and_unit(self.to, other, self.unit)
        if conv is not None:
            nvalue = conv(self.nominal)
            nstd = conv(self.uncertainty)
            return QFloat(nvalue, nstd, unit)
        # None converter means no conversion
        return QFloat(self.nominal, self.uncertainty, self.unit)

    @property
    def value(self) -> _NumberArray:
        """Same as nominal. For Astropy compatibility."""
        return self.nominal

    def __repr__(self) -> str:
        """Identify the quantity by its address and formatted value with uncertainty."""
        i = hex(id(self))
        return f'<QFloat at {i}>\n{self.__str__()}'

    def __str__(self) -> str:
        """Format nominal values and uncertainties with the selected precision and unit."""
        if np.isscalar(self.nominal):
            s = _format_qfloat(self.nominal, self.uncertainty, self.sig_digits)
        else:
            opt = np.get_printoptions()
            a = _create_formater_array(self.nominal, self.uncertainty,
                                       self.sig_digits)
            s = np.array2string(np.array(a), separator=', ',
                                max_line_width=opt['linewidth'],
                                edgeitems=opt['edgeitems'],
                                threshold=50)
        if self.unit != units.dimensionless_unscaled:
            if not np.isscalar(self.nominal):
                s += ' unit='
            else:
                s += ' '
            s += f'{self.unit}'
        return s

    def __getitem__(self, index: _Index) -> QFloat:
        """Get one item of given index IF this is iterable."""
        v = self.nominal[index]
        s = self.uncertainty[index]
        return QFloat(v, s, self.unit)

    def __setitem__(self, index: _Index, value: _QuantityInput) -> None:
        """Set one item at given index if this is iterable."""
        value = convert_to_qfloat(value)
        _, value = same_unit(self, value, self.__setitem__)

        self._nominal[index] = value.nominal
        self._uncert[index] = value.uncertainty

    def __len__(self) -> int:
        """Return the leading nominal-array length; scalar quantities raise TypeError."""
        return len(self.nominal)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: _QuantityInput,
        **kwargs: _t.Any,
    ) -> QFloat | _NumberArray | tuple[QFloat, QFloat] | _types.NotImplementedType:
        """Wrap numpy ufuncs, using uncertainties and units.

        Parameters
        ----------
        function : callable
            Ufunc object that was called.
        method
            String indicating which Ufunc method was called
            (``__call__``, ``reduce``, ``reduceat``, ``accumulate``,
            ``outer`` or ``inner``).
        inputs
            A tuple of the input arguments to the ``ufunc``.
        kwargs
            A dictionary containing the optional input arguments of the
            ``ufunc``. If given, any out arguments, both positional and
            keyword, are passed as a ``tuple`` in kwargs.

        Returns
        -------
        result
            Results of the ufunc, with the unit and uncertainty.
        """
        # Only call supported now
        if method != '__call__':
            return NotImplemented

        if ufunc not in HANDLED_UFUNCS:
            return NotImplemented

        # put all inputs as QFloats, a local "require_qfloat"
        inputs = [convert_to_qfloat(i) for i in inputs]

        out = kwargs.get('out', None)
        if out is not None:
            raise NotImplementedError("`out` argument not supported yet.")

        result = HANDLED_UFUNCS[ufunc](*inputs, **kwargs)

        return result

    def __array_function__(
        self,
        func: _t.Callable[..., object],
        types: _t.Collection[type],
        args: tuple[object, ...],
        kwargs: dict[str, _t.Any],
    ) -> QFloat | _NumberArray | tuple[int, ...] | _types.NotImplementedType:
        """Wrap numpy functions.

        Parameters
        ----------
        func
            Arbitrary callable exposed by NumPy’s public API.
        types
            Collection of unique argument types from the original NumPy
            function call that implement ``__array_function__``.
        args
            Positional arguments directly passed on from the original call.
        kwargs
            Keyword arguments directly passed on from the original call.
        """
        if func not in HANDLED_AFUNCS:
            return NotImplemented

        return HANDLED_AFUNCS[func](*args, **kwargs)

    def __eq__(self, other: object) -> bool | _npt.NDArray[np.bool_]:
        """Test exact equality of all nominal values and uncertainties after unit conversion.

        Return False for incompatible units or objects that cannot be converted.
        This compares the complete quantity rather than returning an elementwise mask.
        """
        try:
            this, other = same_unit(self, other, self.__eq__)
        except Exception:
            # Incompatible units are different numbers.
            # Incompatible types are different
            return False

        if np.any(this.nominal != other.nominal) or \
           np.any(this.uncertainty != other.uncertainty):
            return False
        return True

    def __ne__(self, other: object) -> bool | _npt.NDArray[np.bool_]:
        """Negate exact whole-quantity equality, including uncertainties and unit conversion."""
        return not self == other

    @require_qfloat
    def __gt__(self, other: _QuantityInput) -> bool | _npt.NDArray[np.bool_]:
        """Compare nominal values elementwise after converting to compatible units."""
        this, other = same_unit(self, other, self.__gt__)
        return this.nominal > other.nominal

    @require_qfloat
    def __ge__(self, other: _QuantityInput) -> bool | _npt.NDArray[np.bool_]:
        """Test nominal values for greater-than-or-equal after unit conversion."""
        this, other = same_unit(self, other, self.__lt__)
        return this.nominal >= other.nominal

    @require_qfloat
    def __lt__(self, other: _QuantityInput) -> bool | _npt.NDArray[np.bool_]:
        """Compare nominal values elementwise after converting to compatible units."""
        this, other = same_unit(self, other, self.__lt__)
        return this.nominal < other.nominal

    @require_qfloat
    def __le__(self, other: _QuantityInput) -> bool | _npt.NDArray[np.bool_]:
        """Test nominal values for less-than-or-equal after unit conversion."""
        this, other = same_unit(self, other, self.__lt__)
        return this.nominal <= other.nominal

    def __lshift__(self, other: _UnitLike) -> QFloat:
        """Lshift operator used to convert units."""
        return self.to(other)

    def __ilshift__(self, other: _UnitLike) -> _t.Self:
        """Convert the nominal values and uncertainties to another unit in place."""
        res = self << other
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __add__(self, other: _QuantityInput) -> QFloat:
        """Add quantities, converting compatible units and propagating independent errors."""
        qf1, qf2 = same_unit(self, other, self.__add__)
        sum_n = qf1.nominal + qf2.nominal
        sum_s = propagate_2('add', sum_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(sum_n, sum_s, qf1.unit)

    @require_qfloat
    def __iadd__(self, other: _QuantityInput) -> _t.Self:
        """Add in place, converting units and updating nominal values and errors."""
        res = self.__add__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __radd__(self, other: _QuantityInput) -> QFloat:
        """Add the other operand to this quantity with compatible-unit conversion."""
        return other.__add__(self)

    @require_qfloat
    def __sub__(self, other: _QuantityInput) -> QFloat:
        """Subtract quantities, converting compatible units and propagating independent errors."""
        qf1, qf2 = same_unit(self, other, self.__add__)
        sub_n = qf1.nominal - qf2.nominal
        sub_s = propagate_2('add', sub_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(sub_n, sub_s, qf1.unit)

    @require_qfloat
    def __isub__(self, other: _QuantityInput) -> _t.Self:
        """Subtract in place, converting units and updating nominal values and errors."""
        res = self.__sub__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rsub__(self, other: _QuantityInput) -> QFloat:
        """Subtract this quantity from the other operand with compatible-unit conversion."""
        return -self.__sub__(other)

    @require_qfloat
    def __mul__(self, other: _QuantityInput) -> QFloat:
        """Multiply quantities, combining units and propagating independent errors."""
        unit = self.unit * other.unit
        qf1, qf2 = self, other
        mul_n = qf1.nominal * qf2.nominal
        mul_s = propagate_2('mul', mul_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(mul_n, mul_s, unit)

    @require_qfloat
    def __imul__(self, other: _QuantityInput) -> _t.Self:
        """Multiply in place, updating nominal values, uncertainties and unit."""
        res = self.__mul__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rmul__(self, other: _QuantityInput) -> QFloat:
        """Multiply the other operand by this quantity, combining units and errors."""
        return self.__mul__(other)

    @require_qfloat
    def __truediv__(self, other: _QuantityInput) -> QFloat:
        """Divide quantities, combining units and propagating independent errors."""
        unit = self.unit / other.unit
        qf1, qf2 = self, other
        div_n = qf1.nominal / qf2.nominal
        div_s = propagate_2('truediv', div_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(div_n, div_s, unit)

    @require_qfloat
    def __itruediv__(self, other: _QuantityInput) -> _t.Self:
        """Divide in place, updating nominal values, uncertainties and unit."""
        res = self.__truediv__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rtruediv__(self, other: _QuantityInput) -> QFloat:
        """Divide the other operand by this quantity, combining units and errors."""
        return other.__truediv__(self)

    @require_qfloat
    def __floordiv__(self, other: _QuantityInput) -> QFloat:
        """Floor-divide quantities, using raw nominal values and the quotient unit."""
        unit = self.unit / other.unit
        qf1, qf2 = self, other
        div_n = qf1.nominal // qf2.nominal
        div_s = propagate_2('floordiv', div_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(div_n, div_s, unit)

    @require_qfloat
    def __ifloordiv__(self, other: _QuantityInput) -> _t.Self:
        """Floor-divide in place, updating nominal values, uncertainties and unit."""
        res = self.__floordiv__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rfloordiv__(self, other: _QuantityInput) -> QFloat:
        """Floor-divide the other operand by this quantity using raw nominal values."""
        # As the argument always enter here as a qfloat...
        return other.__floordiv__(self)

    @require_qfloat
    def __mod__(self, other: _QuantityInput) -> QFloat:
        """Take the remainder of quantities, using raw nominal values and retaining the first unit."""
        qf1, qf2 = self, other
        mod_n = qf1.nominal % qf2.nominal
        mod_s = propagate_2('mod', mod_n,
                            qf1.nominal, qf2.nominal,
                            qf1.std_dev, qf2.std_dev)
        return QFloat(mod_n, mod_s, self.unit)

    @require_qfloat
    def __imod__(self, other: _QuantityInput) -> _t.Self:
        """Take the remainder in place, updating nominal values and uncertainties."""
        res = self.__mod__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rmod__(self, other: _QuantityInput) -> QFloat:
        """Take the other operand modulo this quantity using raw nominal values."""
        return other.__mod__(self)

    @require_qfloat
    def __pow__(self, other: _QuantityInput) -> QFloat:
        """Raise the quantity to a dimensionless scalar power and propagate uncertainty.

        The exponent may carry uncertainty, but must be scalar and dimensionless.
        The result unit is the input unit raised to the nominal exponent.
        """
        # Important: different from normal arrays, QFloat cannot be raises
        # to an array due to inconsistencies in unit. Each element could
        # have it's own unit.
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
    def __ipow__(self, other: _QuantityInput) -> _t.Self:
        """Apply exponentiation in place, replacing nominal values, uncertainties and unit."""
        res = self.__pow__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rpow__(self, other: _QuantityInput) -> QFloat:
        """Raise the other operand to this dimensionless scalar exponent."""
        return other.__pow__(self)

    @require_qfloat
    def __neg__(self) -> QFloat:
        """Negate nominal values while preserving standard uncertainties and the unit."""
        return QFloat(-self.nominal, self.uncertainty, self.unit)

    @require_qfloat
    def __pos__(self) -> QFloat:
        """Return a quantity with the same nominal values, uncertainties and unit."""
        return QFloat(self.nominal, self.uncertainty, self.unit)

    @require_qfloat
    def __abs__(self) -> QFloat:
        """Take absolute nominal values while preserving uncertainties and the unit."""
        return QFloat(np.abs(self.nominal), self.uncertainty, self.unit)

    @require_qfloat
    def __int__(self) -> int:
        """Convert the scalar nominal value to int, discarding its unit and uncertainty."""
        return int(self.nominal)

    @require_qfloat
    def __float__(self) -> float:
        """Convert the scalar nominal value to float, discarding its unit and uncertainty."""
        return float(self.nominal)


# TODO:
# Array functions:
#             - copyto, broadcast, broadcast_to
#             - sum, prod, nanprod, nansum, cumprod, cumsum, nancumprod,
#             - nancumsum, diff, ediff1d, cross
#             - concatenate, stack, block, vstack, hstack, dstack, columnstack

@_implements_array_func(np.shape)
def _qfloat_shape(qf: QFloat) -> tuple[int, ...]:
    """Implement np.shape for qfloats."""
    return qf.shape


@_implements_array_func(np.size)
def _qfloat_size(qf: QFloat) -> int:
    """Return the total number of nominal elements for NumPy size dispatch."""
    return qf.size


@_implements_array_func(np.clip)
def _qfloat_clip(
    qf: QFloat,
    a_min: _npt.ArrayLike | None,
    a_max: _npt.ArrayLike | None,
    **kwargs: _t.Any,
) -> QFloat:
    """Clip nominal values while retaining the original uncertainties and unit."""
    # we mantain the original errors
    nominal = np.clip(qf.nominal, a_min, a_max)
    return QFloat(nominal, qf.std_dev, qf.unit)


# Use a simple wrapper for general functions
def _array_func_simple_wrapper(numpy_func: _t.Callable[..., _t.Any]) -> None:
    """Wraps simple array functions.

    Notes
    -----
    - Functions elegible for these are that ones who applies for nominal and
      std_dev values and return a new QFloat with the applied values.
    - No conversion or special treatment is done in this wrapper.
    - Only for one array ate once.
    """
    def wrapper(qf: QFloat, *args: _t.Any, **kwargs: _t.Any) -> QFloat:
        """Apply the same array transformation to nominal values and uncertainties."""
        nominal = numpy_func(qf.nominal, *args, **kwargs)
        std = numpy_func(qf.uncertainty, *args, **kwargs)
        return QFloat(nominal, std, qf.unit)
    _implements_array_func(numpy_func)(wrapper)


_array_func_simple_wrapper(np.delete)
_array_func_simple_wrapper(np.expand_dims)
_array_func_simple_wrapper(np.flip)
_array_func_simple_wrapper(np.fliplr)
_array_func_simple_wrapper(np.flipud)
_array_func_simple_wrapper(np.moveaxis)
_array_func_simple_wrapper(np.ravel)
_array_func_simple_wrapper(np.repeat)
_array_func_simple_wrapper(np.reshape)
_array_func_simple_wrapper(np.resize)
_array_func_simple_wrapper(np.roll)
_array_func_simple_wrapper(np.rollaxis)
_array_func_simple_wrapper(np.rot90)
_array_func_simple_wrapper(np.squeeze)
_array_func_simple_wrapper(np.swapaxes)
_array_func_simple_wrapper(np.take)
_array_func_simple_wrapper(np.tile)
_array_func_simple_wrapper(np.transpose)


@_implements_array_func(np.round)
@_implements_array_func(np.around)
def _qfloat_round(qf: QFloat, decimals: int=0) -> QFloat:
    """Implement np.round for qfloats."""
    nominal = np.round(qf.nominal, decimals)
    std = np.round(qf.uncertainty, decimals)
    return QFloat(nominal, std, qf.unit)


@_implements_array_func(np.append)
def _qfloat_append(qf: QFloat, values: _QuantityInput, axis: _Axis=None) -> QFloat:
    """Implement np.append for qfloats."""
    # First, convert to the same unit.
    qf1, qf2 = same_unit(qf, values, func=np.append)
    nominal = np.append(qf1.nominal, qf2.nominal, axis)
    std = np.append(qf1.uncertainty, qf2.uncertainty, axis)
    return QFloat(nominal, std, qf1.unit)


@_implements_array_func(np.insert)
def _qfloat_insert(
    qf: QFloat,
    obj: _Index,
    values: _QuantityInput,
    axis: _Axis=None,
) -> QFloat:
    """Implement np.insert for qfloats."""
    # Same unit needed too
    qf1, qf2 = same_unit(qf, values, func=np.insert)
    nominal = np.insert(qf1.nominal, obj, qf2.nominal, axis)
    std = np.insert(qf1.uncertainty, obj, qf2.uncertainty, axis)
    return QFloat(nominal, std, qf1.unit)


@_implements_array_func(np.sum)
def _qfloat_sum(qf: QFloat, axis: _Axis=None) -> QFloat:
    """Implement np.sum for qfloats."""
    nominal = np.sum(qf.nominal, axis=axis)
    std = np.sqrt(np.sum(np.square(qf.uncertainty), axis=axis))
    return QFloat(nominal, std, qf.unit)


@_implements_array_func(np.mean)
def _qfloat_mean(qf: QFloat, axis: _Axis=None) -> QFloat:
    """Implement np.mean for qfloats."""
    nominal = np.mean(qf.nominal, axis=axis)
    # error of average = std_dev/sqrt(N)
    std = np.std(qf.nominal, axis=axis)
    # N is determined by the number of elements in the axis
    std /= np.sqrt(np.sum(np.ones_like(qf.nominal), axis=axis))
    return QFloat(nominal, std, qf.unit)


@_implements_array_func(np.nanmean)
def _qfloat_nanmean(qf: QFloat, axis: _Axis=None) -> QFloat:
    """Implement np.mean for qfloats."""
    nominal = np.nanmean(qf.nominal, axis=axis)
    # error of average = std_dev/sqrt(N)
    std = np.nanstd(qf.nominal, axis=axis)
    # N is determined by the number of elements in the axis
    std /= np.sqrt(np.nansum(qf.nominal, axis=axis)/nominal)
    return QFloat(nominal, std, qf.unit)


@_implements_array_func(np.median)
def _qfloat_median(qf: QFloat, axis: _Axis=None) -> QFloat:
    """Implement np.median for qfloats."""
    nominal = np.median(qf.nominal, axis=axis)
    # error of median = error of average = std_dev/sqrt(N)
    std = np.nanstd(qf.nominal, axis=axis)
    # N is determined by the number of elements in the axis
    std /= np.sqrt(np.sum(np.ones_like(qf.nominal), axis=axis))
    return QFloat(nominal, std, qf.unit)


@_implements_array_func(np.nanmedian)
def _qfloat_nanmedian(qf: QFloat, axis: _Axis=None) -> QFloat:
    """Implement np.median for qfloats."""
    nominal = np.nanmedian(qf.nominal, axis=axis)
    # error of average = std_dev/sqrt(N)
    std = np.nanstd(qf.nominal, axis=axis)
    # N is determined by the number of elements in the axis
    std /= np.sqrt(np.nansum(qf.nominal, axis=axis)/nominal)
    return QFloat(nominal, std, qf.unit)


@_implements_array_func(np.std)
def _qfloat_std(qf: QFloat, axis: _Axis=None) -> _NumberArray:
    """Implement np.std for qfloats."""
    return np.std(qf.nominal, axis=axis)


@_implements_array_func(np.nanstd)
def _qfloat_nanstd(qf: QFloat, axis: _Axis=None) -> _NumberArray:
    """Implement np.std for qfloats."""
    return np.nanstd(qf.nominal, axis=axis)


@_implements_array_func(np.var)
def _qfloat_var(qf: QFloat, axis: _Axis=None) -> _NumberArray:
    """Implement np.var for qfloats."""
    return np.var(qf.nominal, axis=axis)


@_implements_array_func(np.nanvar)
def _qfloat_nanvar(qf: QFloat, axis: _Axis=None) -> _NumberArray:
    """Implement np.var for qfloats."""
    return np.nanvar(qf.nominal, axis=axis)


def _implements_ufunc_on_nominal(func: np.ufunc) -> None:
    """Wraps ufuncs only on the nominal value and don't return QFloat."""
    def wrapper(qf: QFloat, *args: _t.Any, **kwargs: _t.Any) -> _NumberArray:
        """Apply the ufunc to nominal values and return its ordinary NumPy result."""
        return func(qf.nominal, *args, **kwargs)
    _implements_ufunc(func)(wrapper)


_implements_ufunc_on_nominal(np.isnan)
_implements_ufunc_on_nominal(np.isinf)
_implements_ufunc_on_nominal(np.isfinite)
_implements_ufunc_on_nominal(np.isneginf)
_implements_ufunc_on_nominal(np.isposinf)
_implements_ufunc_on_nominal(np.isreal)
_implements_ufunc_on_nominal(np.iscomplex)
_implements_ufunc_on_nominal(np.isscalar)
_implements_ufunc_on_nominal(np.signbit)
_implements_ufunc_on_nominal(np.sign)


def _qfloat_exp_log(qf: QFloat, func: np.ufunc) -> QFloat:
    """General implementation for exp and log functions."""
    if qf.unit != units.dimensionless_unscaled:
        raise UnitsError(f'{func.__name__} is only defined for dimensionless'
                         ' quantities.')
    val = func(qf.nominal)
    std = propagate_1(func.__name__, val, qf.nominal, qf.uncertainty)
    return QFloat(val, std, units.dimensionless_unscaled)


_implements_ufunc(np.exp)(partial(_qfloat_exp_log, func=np.exp))
_implements_ufunc(np.exp2)(partial(_qfloat_exp_log, func=np.exp2))
_implements_ufunc(np.expm1)(partial(_qfloat_exp_log, func=np.expm1))
_implements_ufunc(np.log)(partial(_qfloat_exp_log, func=np.log))
_implements_ufunc(np.log2)(partial(_qfloat_exp_log, func=np.log2))
_implements_ufunc(np.log10)(partial(_qfloat_exp_log, func=np.log10))
_implements_ufunc(np.log1p)(partial(_qfloat_exp_log, func=np.log1p))


def _qfloat_floor_wrapper(qf: QFloat, func: np.ufunc) -> QFloat:
    """General implementation for floor, ceil and trunc."""
    return QFloat(func(qf.nominal), np.round(qf.uncertainty, 0), qf.unit)


_implements_ufunc(np.floor)(partial(_qfloat_floor_wrapper, func=np.floor))
_implements_ufunc(np.ceil)(partial(_qfloat_floor_wrapper, func=np.ceil))
_implements_ufunc(np.trunc)(partial(_qfloat_floor_wrapper, func=np.trunc))


def _qfloat_only_nominal_wrapper(qf: QFloat, func: np.ufunc) -> _NumberArray:
    """General implementation for isfinite, isinf, isnan."""
    return func(qf.nominal)


_implements_ufunc(np.isfinite)(partial(_qfloat_only_nominal_wrapper,
                                       func=np.isfinite))
_implements_ufunc(np.isinf)(partial(_qfloat_only_nominal_wrapper,
                                    func=np.isinf))
_implements_ufunc(np.isnan)(partial(_qfloat_only_nominal_wrapper,
                                    func=np.isnan))


@_implements_ufunc(np.radians)
@_implements_ufunc(np.deg2rad)
def _qfloat_radians(qf: QFloat, *args: _t.Any, **kwargs: _t.Any) -> QFloat:
    """Convert any qfloat angle to radian."""
    return qf.to(units.radian)


@_implements_ufunc(np.degrees)
@_implements_ufunc(np.rad2deg)
def _qfloat_degrees(qf: QFloat, *args: _t.Any, **kwargs: _t.Any) -> QFloat:
    """Convert angular nominal values and uncertainties to degrees."""
    return qf.to(units.degree)


def _trigonometric_simple_wrapper(numpy_ufunc: np.ufunc) -> None:
    """Register an angular ufunc with derivative-based uncertainty propagation."""
    def trig_wrapper(qf: QFloat, *args: _t.Any, **kwargs: _t.Any) -> QFloat:
        """Evaluate an angle in radians and return dimensionless values with errors."""
        # check if qf is angle
        if qf.unit not in (units.degree, units.radian):
            raise UnitsError('qfloat unit is not degree or radian.')

        # if degree, convert to radian as required for numpy inputs.
        if qf.unit == units.degree:
            qf = qf.to(units.radian)

        nominal = numpy_ufunc(qf.nominal)
        std = propagate_1(numpy_ufunc.__name__, nominal,
                          qf.nominal, qf.std_dev)
        return QFloat(nominal, std, units.dimensionless_unscaled)
    _implements_ufunc(numpy_ufunc)(trig_wrapper)


def _inverse_trigonometric_simple_wrapper(numpy_ufunc: np.ufunc) -> None:
    """Register an inverse angular ufunc for dimensionless inputs."""
    def inv_wrapper(qf: QFloat, *args: _t.Any, **kwargs: _t.Any) -> QFloat:
        """Evaluate a dimensionless input and return radians with propagated errors."""
        if qf.unit != units.dimensionless_unscaled:
            raise UnitsError('inverse trigonometric functions require '
                             'dimensionless unscaled variables.')

        nominal = numpy_ufunc(qf.nominal)
        std = propagate_1(numpy_ufunc.__name__, nominal,
                          qf.nominal, qf.std_dev)

        return QFloat(nominal, std, units.radian)
    _implements_ufunc(numpy_ufunc)(inv_wrapper)


_trigonometric_simple_wrapper(np.sin)
_trigonometric_simple_wrapper(np.cos)
_trigonometric_simple_wrapper(np.tan)
_trigonometric_simple_wrapper(np.sinh)
_trigonometric_simple_wrapper(np.cosh)
_trigonometric_simple_wrapper(np.tanh)
_inverse_trigonometric_simple_wrapper(np.arcsin)
_inverse_trigonometric_simple_wrapper(np.arccos)
_inverse_trigonometric_simple_wrapper(np.arctan)
_inverse_trigonometric_simple_wrapper(np.arcsinh)
_inverse_trigonometric_simple_wrapper(np.arccosh)
_inverse_trigonometric_simple_wrapper(np.arctanh)


@_implements_ufunc(np.arctan2)
def _qfloat_arctan2(qf1: QFloat, qf2: QFloat) -> QFloat:
    """Compute the arctangent of qf1/qf2."""
    # The 2 values must be in the same unit.
    qf2 = qf2.to(qf1.unit)
    nominal = np.arctan2(qf1.nominal, qf2.nominal)
    std = propagate_2('arctan2', nominal, qf1.nominal, qf2.nominal,
                      qf1.std_dev, qf2.std_dev)
    return QFloat(nominal, std, units.radian)


_ufunc_translate = {
    'add': QFloat.__add__,
    'absolute': QFloat.__abs__,
    'divide': QFloat.__truediv__,
    'float_power': QFloat.__pow__,
    'floor_divide': QFloat.__floordiv__,
    'multiply': QFloat.__mul__,
    'negative': QFloat.__neg__,
    'positive': QFloat.__pos__,
    'power': QFloat.__pow__,
    'mod': QFloat.__mod__,
    'remainder': QFloat.__mod__,
    'subtract': QFloat.__sub__,
    'true_divide': QFloat.__truediv__,
    'divmod': lambda x, y: (QFloat.__floordiv__(x, y), QFloat.__mod__(x, y)),
}


def _general_ufunc_wrapper(numpy_ufunc: np.ufunc) -> None:
    """Implement ufuncs for general math operations.

    Notes
    -----
    - These functions will not operate with kwarg.
    - These functions will just wrap QFloat math methods.
    """
    ufunc_name = numpy_ufunc.__name__
    true_func = _ufunc_translate[ufunc_name]

    def ufunc_wrapper(*inputs: QFloat) -> QFloat | tuple[QFloat, QFloat]:
        """Dispatch converted operands to the corresponding QFloat arithmetic method."""
        return true_func(*inputs)
    _implements_ufunc(numpy_ufunc)(ufunc_wrapper)


_general_ufunc_wrapper(np.add)
_general_ufunc_wrapper(np.absolute)
_general_ufunc_wrapper(np.divide)
_general_ufunc_wrapper(np.divmod)
_general_ufunc_wrapper(np.float_power)
_general_ufunc_wrapper(np.floor_divide)
_general_ufunc_wrapper(np.mod)
_general_ufunc_wrapper(np.multiply)
_general_ufunc_wrapper(np.negative)
_general_ufunc_wrapper(np.positive)
_general_ufunc_wrapper(np.power)
_general_ufunc_wrapper(np.remainder)
_general_ufunc_wrapper(np.subtract)
_general_ufunc_wrapper(np.true_divide)


@_implements_ufunc(np.copysign)
def _qfloat_copysign(qf1: QFloat, qf2: QFloat) -> QFloat:
    """Return the first argument with the sign of the second argument."""
    nominal = np.copysign(qf1.nominal, qf2.nominal)
    std = propagate_2('copysign', nominal, qf1.nominal, qf2.nominal,
                      qf1.std_dev, qf2.std_dev)
    return QFloat(nominal, std, qf1.unit)


@_implements_ufunc(np.square)
def _qfloat_square(qf: QFloat) -> QFloat:
    """Square the quantity through QFloat multiplication and error propagation."""
    return qf * qf


@_implements_ufunc(np.sqrt)
def _qfloat_sqrt(qf: QFloat) -> QFloat:
    """Take the square root through QFloat exponentiation and error propagation."""
    return qf**0.5


@_implements_ufunc(np.hypot)
def _qfloat_hypot(qf1: QFloat, qf2: QFloat) -> QFloat:
    """Compute a hypotenuse in the first operand unit with independent-error propagation."""
    qf2 = qf2.to(qf1.unit)
    nominal = np.hypot(qf1.nominal, qf2.nominal)
    std = propagate_2('hypot', nominal, qf1.nominal, qf2.nominal,
                      qf1.std_dev, qf2.std_dev)
    return QFloat(nominal, std, qf1.unit)
