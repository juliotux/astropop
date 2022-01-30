# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Put a `unit` property in classes."""

from astropy import units


def unit_property(cls):
    """Add a `unit` property to a class."""
    def _unit_getter(obj):
        if obj._unit is None:
            return units.dimensionless_unscaled
        return obj._unit

    def _unit_setter(obj, value):
        if value is None or units.Unit(value) == units.dimensionless_unscaled:
            obj._unit = None
        else:
            obj._unit = units.Unit(value)

    cls._unit = None
    cls.unit = property(_unit_getter, _unit_setter,
                        doc="Physical unit of the data.")
    return cls
