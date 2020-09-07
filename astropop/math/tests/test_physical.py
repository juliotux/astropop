# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from astropop.math.physical import UFloat, ufloat, unit_property, units
import pytest_check as check


@unit_property
class DummyClass():
    def __init__(self, unit):
        self.unit = unit


@pytest.mark.parametrize('unit,expect', [('meter', units.m),
                                         (units.adu, units.adu),
                                         (None, units.dimensionless_unscaled),
                                         ('', units.dimensionless_unscaled)])
def test_unit_property(unit, expect):
    # Getter test
    c = DummyClass(unit)
    check.equal(c.unit, expect)

    # Setter test
    c = DummyClass(None)
    c.unit = unit
    check.equal(c.unit, expect)


@pytest.mark.parametrize('unit', ['', units.dimensionless_unscaled, None])
def test_unit_property_none(unit):
    # Check None and dimensionless_unscaled
    c = DummyClass(unit)
    check.is_none(c._unit)


@pytest.mark.parametrize('unit', ['None', 'Invalid'])
def test_unit_property_invalid(unit):
    with pytest.raises(ValueError):
        c = DummyClass(unit)


def test_ufloat_creation():
    def _create(*args, **kwargs):
        check.is_instance(ufloat(*args, **kwargs), UFloat)

    def _raises(*args, **kwargs):
        with pytest.raises(Exception):
            ufloat(*args, **kwargs)

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
