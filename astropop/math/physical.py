# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Math operations with uncertainties and units.

Simplified version of `uncertainties` python package with some
`~astropy.units` addings, in a much more free form.
"""

# TODO: Numpy ufuncs compatibility
# TODO: Numpy array funcs compatibility

import numbers
from astropy import units
from astropy.units.quantity_helper.helpers import get_converters_and_unit
from astropy.units import UnitsError, Quantity
import numpy as np
from uncertainties import unumpy as unp
from uncertainties import ufloat, UFloat

from ..py_utils import check_iterable


__all__ = ['unit_property', 'QFloat', 'qfloat', 'units', 'UnitsError',
           'equal_within_errors']


HANDLED_FUNCTIONS = {}


def implements(numpy_function):
    """Register an __array_function__ implementation for QFloat objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


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


def ufloat_or_uarray(qf):
    """Convert a qfloat to a ufloat, according to iterability."""
    if check_iterable(qf.nominal):
        return unp.uarray(qf.nominal, qf.std_dev)
    return ufloat(qf.nominal, qf.std_dev)


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

    # UFloat support
    if isinstance(value, UFloat):
        return QFloat(value.n, value.s, unit)

    # Numpy arrays support. They need to handle single numbers or ufloat
    if isinstance(value, (np.ndarray, list, tuple)):
        # UFloat members
        if isinstance(np.ravel(value)[0], UFloat):
            return QFloat(unp.nominal_values(value), unp.std_devs(value), unit)
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
    """Create a UFloat from the values.

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

        self._uncert = None  # always value is reset, uncertainty resets
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
        nvalue = conv(self.nominal)
        nstd = conv(self.uncertainty)
        return QFloat(nvalue, nstd, unit)

    def __repr__(self):
        # repr for arrays
        if check_iterable(self.nominal):
            ret = "<QFloat\n"
            ret2 = unp.uarray(self.nominal, self.uncertainty).__repr__()
            ret2 += f'\n      {str(self.unit)}'
        # repr for single values
        else:
            ret = "<QFloat "
            ret2 = ufloat(self.nominal, self.uncertainty).__repr__()
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
        """
        raise NotImplementedError

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
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented

        if not all(issubclass(t, QFloat) for t in types):
            return NotImplemented

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

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
        this, other = same_unit(self, other, self.__add__)
        uf1 = ufloat_or_uarray(this)
        uf2 = ufloat_or_uarray(other)
        res = uf1 + uf2
        if isinstance(res, np.ndarray):
            nominal = unp.nominal_values(res)
            std_dev = unp.std_devs(res)
        else:
            nominal = res.nominal_value
            std_dev = res.std_dev

        return QFloat(nominal, std_dev, this.unit)

    @require_qfloat
    def __iadd__(self, other):
        res = self.__add__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __radd__(self, other):
        return self.__add__(other)

    @require_qfloat
    def __sub__(self, other):
        this, other = same_unit(self, other, self.__add__)
        uf1 = ufloat_or_uarray(this)
        uf2 = ufloat_or_uarray(other)
        res = uf1 - uf2
        if isinstance(res, np.ndarray):
            nominal = unp.nominal_values(res)
            std_dev = unp.std_devs(res)
        else:
            nominal = res.nominal_value
            std_dev = res.std_dev

        return QFloat(nominal, std_dev, this.unit)

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
        uf1 = ufloat_or_uarray(self)
        uf2 = ufloat_or_uarray(other)
        res = uf1 * uf2
        if isinstance(res, np.ndarray):
            nominal = unp.nominal_values(res)
            std_dev = unp.std_devs(res)
        else:
            nominal = res.nominal_value
            std_dev = res.std_dev
        return QFloat(nominal, std_dev, unit)

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
        raise NotImplementedError

    @require_qfloat
    def __itruediv__(self, other):
        res = self.__truediv__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rtruediv__(self, other):
        raise NotImplementedError

    @require_qfloat
    def __floordiv__(self, other):
        raise NotImplementedError

    @require_qfloat
    def __ifloordiv__(self, other):
        res = self.__floordiv__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rfloordiv__(self, other):
        raise NotImplementedError

    @require_qfloat
    def __div__(self, other):
        return self.__truediv__(other)

    @require_qfloat
    def __idiv__(self, other):
        return self.__itruediv__(other)

    @require_qfloat
    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    @require_qfloat
    def __mod__(self, other):
        raise NotImplementedError

    @require_qfloat
    def __imod__(self, other):
        res = self.__mod__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rmod__(self, other):
        raise NotImplementedError

    @require_qfloat
    def __pow__(self, other):
        raise NotImplementedError

    @require_qfloat
    def __ipow__(self, other):
        res = self.__pow__(other)
        self.reset(res.nominal, res.uncertainty, res.unit)
        return self

    @require_qfloat
    def __rpow__(self, other):
        raise NotImplementedError

    @require_qfloat
    def __neg__(self):
        return QFloat(-self.nominal, self.uncertainty, self.unit)

    @require_qfloat
    def __pos__(self):
        return QFloat(self.nominal, self.uncertainty, self.unit)

    @require_qfloat
    def __abs__(self):
        raise NotImplementedError

    @require_qfloat
    def __invert__(self):
        raise NotImplementedError

    @require_qfloat
    def __int__(self):
        raise NotImplementedError

    @require_qfloat
    def __float__(self):
        raise NotImplementedError

    @require_qfloat
    def __round__(self):
        raise NotImplementedError

    @require_qfloat
    def __trunc__(self):
        raise NotImplementedError

    @require_qfloat
    def __floor__(self):
        raise NotImplementedError

    @require_qfloat
    def __ceil__(self):
        raise NotImplementedError
