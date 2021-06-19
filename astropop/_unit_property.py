# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Put a `unit` property in classes."""

from astropy import units


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
