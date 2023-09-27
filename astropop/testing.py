# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Collection of testing helpers.

Notes
-----
- Based on pytest_check. However, we are removing the dependency.
- Lots of functions imported directly from numpy.testing
"""

from os.path import exists
import numpy as np
from numpy import testing as npt


__all__ = ['assert_equal', 'assert_not_equal', 'assert_almost_equal',
           'assert_true', 'assert_false',
           'assert_is', 'assert_is_not',
           'assert_is_none', 'assert_is_not_none',
           'assert_in', 'assert_not_in',
           'assert_is_instance', 'assert_is_not_instance',
           'assert_greater', 'assert_greater_equal',
           'assert_less', 'assert_less_equal',
           'assert_raises', 'assert_raises_regex',
           'assert_warns', 'assert_not_warnings',
           'assert_path_exists', 'assert_path_not_exists']


def func_wrapper(func):
    """Warp the function, if needed. Now, its dummy."""
    # The idea here is to, in future, continue the tests and log all
    # failures at once. Like pytest_check.
    return func


# --------------------------------------------------------------------
# Testing Helpers
# --------------------------------------------------------------------

assert_raises = func_wrapper(npt.assert_raises)
assert_raises_regex = func_wrapper(npt.assert_raises_regex)
assert_warns = func_wrapper(npt.assert_warns)
assert_not_warnings = func_wrapper(npt.assert_no_warnings)


@func_wrapper
def assert_true(a, msg=None):
    """Raise assertion error if a is not True."""
    if not a:
        msg = msg or f"{a} is not True."
        raise AssertionError(msg)


@func_wrapper
def assert_false(a, msg=None):
    """Raise assertion error if condition is not False."""
    if a:
        msg = msg or f"{a} is not False."
        raise AssertionError(msg)


@func_wrapper
def assert_equal(a, b, msg=None):
    """Check if two objects are equal. Arrays supported."""
    if not np.isscalar(a) or not np.isscalar(b):
        if msg is None:
            msg = ''
        npt.assert_array_equal(a, b, err_msg=msg, verbose=True)
    else:
        if not a == b:
            msg = msg or f"{a} is not equal to {b}."
            raise AssertionError(msg)


@func_wrapper
def assert_almost_equal(a, b, decimal=6, msg=None):
    """Check if two objects are almost equal. Arrays supported."""
    if not np.isscalar(a) or not np.isscalar(b):
        if msg is None:
            msg = ''
        npt.assert_array_almost_equal(a, b, decimal=decimal,
                                      err_msg=msg, verbose=True)
    else:
        limit = 1.5 * 10**(-decimal)
        if not abs(a-b) < limit:
            msg = msg or f"{a} is not almost equal {b} within {limit}."
            raise AssertionError(msg)


@func_wrapper
def assert_not_equal(a, b, msg=None):
    """Raise assertion error if values are equal."""
    try:
        npt.assert_equal(a, b)
    except AssertionError:
        pass
    else:
        msg = msg or f"{a} is equal to {b}."
        raise AssertionError(msg)


@func_wrapper
def assert_is(a, b, msg=None):
    """Raise assertion error if `a is b`."""
    if a is not b:
        msg = msg or f"{a} is not {b}."
        raise AssertionError(msg)


@func_wrapper
def assert_is_not(a, b, msg=None):
    """Raise assertion error if `a is not b`."""
    if a is b:
        msg = msg or f"{a} is {b}."
        raise AssertionError(msg)


@func_wrapper
def assert_is_none(a, msg=None):
    """Raise assertion error if `a is not None`."""
    if a is not None:
        msg = msg or f"{a} is not None."
        raise AssertionError(msg)


@func_wrapper
def assert_is_not_none(a, msg=None):
    """Raise assertion error if `a is None`."""
    if a is None:
        msg = msg or f"{a} is None."
        raise AssertionError(msg)


@func_wrapper
def assert_in(a, b, msg=None):
    """Raise assertion error if `a not in b`."""
    if a not in b:
        msg = msg or f"{a} is not in {b}."
        raise AssertionError(msg)


@func_wrapper
def assert_not_in(a, b, msg=None):
    """Raise assertion error if `a in b`."""
    if a in b:
        msg = msg or f"{a} is in {b}."
        raise AssertionError(msg)


@func_wrapper
def assert_is_instance(a, b, msg=None):
    """Raise assertion error if not `isinstance(a, b)`."""
    if not isinstance(a, b):
        msg = msg or f"{a} is not instance of {b}."
        raise AssertionError(msg)


@func_wrapper
def assert_is_not_instance(a, b, msg=None):
    """Raise assertion error if `isinstance(a, b)`."""
    if isinstance(a, b):
        msg = msg or f"{a} is instance of {b}."
        raise AssertionError(msg)


@func_wrapper
def assert_greater(a, b, msg=None):
    """Raise assertion error if a is not greater then b."""
    if not a > b:
        msg = msg or f"{a} is not greater than {b}."
        raise AssertionError(msg)


@func_wrapper
def assert_greater_equal(a, b, msg=None):
    """Raise assertion error if a is not greater or equal then b."""
    if not a >= b:
        msg = msg or f"{a} is not greater or equal than {b}."
        raise AssertionError(msg)


@func_wrapper
def assert_less(a, b, msg=None):
    """Raise assertion error if a is not less b."""
    if not a < b:
        msg = msg or f"{a} is not less than {b}."
        raise AssertionError(msg)


@func_wrapper
def assert_less_equal(a, b, msg=None):
    """Raise assertion error if a is not less or equal then b."""
    if not a <= b:
        msg = msg or f"{a} is not less or equal than {b}."
        raise AssertionError(msg)


@func_wrapper
def assert_path_exists(a, msg=None):
    """Raise assertion error if path a do not exists."""
    if not exists(a):
        msg = msg or f"{a} do not exists."
        raise AssertionError(msg)


@func_wrapper
def assert_path_not_exists(a, msg=None):
    """Raise assertion error if path a do exists."""
    if exists(a):
        msg = msg or f"{a} do exists."
        raise AssertionError(msg)
