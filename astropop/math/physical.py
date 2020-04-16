# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Math operations with uncertainties and units.

Simplified version of `uncertainties` python package with some
`~astropy.units` addings, in a much more free form."""


from astropy import units
import numpy as np

from ..py_utils import check_iterable


__all__ = ['unit_property', 'UFloat', 'ufloat', 'units']


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
    """Storing float values with uncertainties and units.

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

    # TODO: Math operators
    # TODO: Array wrappers

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
