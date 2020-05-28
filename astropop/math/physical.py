# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Math operations with uncertainties and units.

Simplified version of `uncertainties` python package with some
`~astropy.units` addings, in a much more free form."""

import copy
from astropy import units
import numpy as np

from ._deriv import numpy_ufunc_derivatives, math_derivatives
from ..py_utils import check_iterable
from ..logger import logger


__all__ = ['unit_property', 'UFloat', 'ufloat', 'units']

# pylint:disable=no-else-return,no-else-raise


def _filter_compatible(inp, cls, attr, else_None=False):
    """Filter common data structures compatible with UFloat."""
    if else_None:
        inp = tuple(getattr(x, attr) if isinstance(x, cls) else None
                    for x in inp)
    else:
        inp = tuple(getattr(x, attr) if isinstance(x, cls) else x
                    for x in inp)
    return inp


def unit_property(cls):
    """Add a `unit` property to a class."""
    def _unit_getter(self):
        if self._unit is None:  # noqa:W0212
            return units.dimensionless_unscaled
        return self._unit  # noqa:W0212

    def _unit_setter(self, value):
        if value is None or units.Unit(value) == units.dimensionless_unscaled:
            self._unit = None  # noqa:W0212
        else:
            self._unit = units.Unit(value)  # noqa:W0212

    cls._unit = None  # noqa:W0212
    cls.unit = property(_unit_getter, _unit_setter,
                        doc="Physical unit of the data.")
    return cls


@unit_property
class UFloat():
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
            self._uncert = None
        else:
            if np.shape(value) != np.shape(self._nominal):
                raise ValueError('Uncertainty with shape different from '
                                 'nominal value: '
                                 f'{np.shape(value)} '
                                 f'{np.shape(self._nominal)}')
            if check_iterable(self._nominal):
                self._uncert = np.array(value)
            else:
                self._uncert = float(value)

    def _set_nominal(self, value):
        if value is None:
            raise ValueError('Nominal value cannot be None')
        self._nominal = value
        self._uncert = None  # always value is reset, uncertainty resets
        # No unit changes

    @property
    def uncertainty(self):
        """Uncertainty of the quantity."""
        if self._uncert is None:
            if check_iterable(self._nominal):
                return np.zeros_like(self._nominal)
            else:
                return 0.0
        else:
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

    def __repr__(self):
        ret = "< UFloat "
        if check_iterable(self._nominal):
            ret += str(np.shape(self._nominal))
        else:
            ret += str(self._nominal)
            if self._uncert is not None:
                ret += f"+-{self._uncert}"
        ret += f" {self.unit} "
        ret += " >"
        return ret

    def _compute_errors(self, derivs, inpnom, inpstd, **kwargs):
        """Compute the error components using func and derivatives."""
        n_derivs = len(derivs)  # number of expected numerical inputs?
        # check if the number of inputs matches the number of derivs
        if len(inpnom) != n_derivs or len(inpstd) != n_derivs:
            raise ValueError('Inputs and derivatives have different number '
                             'of components')

        axis = kwargs.get('axis')
        if axis:
            raise NotImplementedError('Not implemented for apply in axis.')
        else:
            components = [None]*n_derivs
            for i in range(n_derivs):
                components[i] = derivs[i](*inpnom)*inpstd[i]
            return np.sqrt(np.sum(np.square(components)))
        return None

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # TODO: check units across the inputs (including inside lists)
        global logger

        inpnom = copy.copy(inputs)
        for c, a in zip([UFloat], ['nominal']):
            # This allows more customization
            inpnom = _filter_compatible(inputs, c, a)

        inpstd = copy.copy(inputs)
        for c, a in zip([UFloat], ['uncertainty']):
            # This allows more customization
            inpstd = _filter_compatible(inputs, c, a, else_None=True)

        nkwargs = copy.copy(kwargs)
        skwargs = copy.copy(kwargs)
        if kwargs.get('out', ()):
            nkwargs['out'] = _filter_compatible(nkwargs['out'],
                                                UFloat, 'nominal')
            skwargs['out'] = _filter_compatible(skwargs['out'],
                                                UFloat, 'uncertainty',
                                                else_None=True)

        ufn = ufunc.__name__
        nominal = getattr(ufunc, method)(*inpnom, **nkwargs)
        if ufn in numpy_ufunc_derivatives:
            std_func = numpy_ufunc_derivatives[ufn]
            std = self._compute_errors(std_func, inpnom, inpstd, **skwargs)
        else:
            logger.warning("Function %s errors is not implemented.", ufn)
            std = None

        if isinstance(nominal, tuple):
            if std is None:
                std = [None]*len(nominal)
            return tuple(UFloat(n, s, self.unit)
                         for n, s in zip(nominal, std))
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return UFloat(nominal, std, self.unit)


def ufloat(value, uncertainty=None, unit=None):
    """Create a UFloat quantity to handle operations. Just wrap UFloat

    Parameters
    ----------
    value : number or array_like
        Nominal value(s) of the quantity.
    uncertainty : number, array_like or `None` (optional)
        Uncertainty value of the quantity. If `None`, the quantity will be
        considered with no errors. Must match `value` shape.
    unit : `~astropy.units.Unit` or string (optional)
        The data unit. Must be `~astropy.units.Unit` compliant.

    Returns
    -------
    q : `UFloat`
        Quantity generated value, with uncertainty and unit.
    """
    q = UFloat(value, uncertainty, unit)
    return q
