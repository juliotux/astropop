# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
from astropop.math.physical import units
from astropop._unit_property import unit_property
from astropop.testing import *


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
    assert_equal(c.unit, expect)

    # Setter test
    c = DummyClass(None)
    c.unit = unit
    assert_equal(c.unit, expect)


def test_qfloat_unit_property_none():
    # Check None and dimensionless_unscaled
    c = DummyClass(None)
    assert_is_none(c._unit)
    assert_equal(c.unit, units.dimensionless_unscaled)


@pytest.mark.parametrize('unit', ['None', 'Invalid'])
def test_qfloat_unit_property_invalid(unit):
    with pytest.raises(ValueError):
        DummyClass(unit)
