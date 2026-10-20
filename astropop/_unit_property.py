# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Put a ``unit`` property in classes."""

from __future__ import annotations

import typing as _t



from astropy import units

_C = _t.TypeVar('_C', bound=type)


class _UnitContainer(_t.Protocol):
    """Describe the internal unit storage used by the unit descriptor."""
    _unit: units.UnitBase | None


def unit_property(cls: _C) -> _C:
    """Add a ``unit`` property to a class."""
    def _unit_getter(obj: _UnitContainer) -> units.UnitBase:
        """Return the stored unit, treating missing storage as dimensionless."""
        if obj._unit is None:
            return units.dimensionless_unscaled
        return obj._unit

    def _unit_setter(
        obj: _UnitContainer,
        value: str | units.UnitBase | units.Quantity | None,
    ) -> None:
        """Normalize the unit, storing dimensionless values internally as None."""
        if value is None or units.Unit(value) == units.dimensionless_unscaled:
            obj._unit = None
        else:
            obj._unit = units.Unit(value)

    cls._unit = None
    cls.unit = property(_unit_getter, _unit_setter,
                        doc="Physical unit of the data.")
    return cls
