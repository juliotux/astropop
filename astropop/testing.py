# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Collection of testing helpers.

Notes
-----
- Based on pytest_check. However, we are removing the dependency.
- Lots of functions imported directly from numpy.testing
"""

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
           'assert_warns', 'assert_not_warnings']


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
def assert_true(a, msg=''):
    """Raise assertion error if a is not True."""
    assert a, msg


@func_wrapper
def assert_false(a, msg=''):
    """Raise assertion error if condition is not False."""
    assert not a, msg


assert_equal = func_wrapper(npt.assert_array_equal)
assert_almost_equal = func_wrapper(npt.assert_array_almost_equal)


@func_wrapper
def assert_not_equal(a, b, msg=''):
    """Raise assertion error if values are equal."""
    try:
        npt.assert_equal(a, b)
    except AssertionError:
        pass
    else:
        raise AssertionError(msg)


@func_wrapper
def assert_is(a, b, msg=''):
    """Raise assertion error if a is not b."""
    assert a is b, msg


@func_wrapper
def assert_is_not(a, b, msg=''):
    """Raise assertion error if a is b."""
    assert a is not b, msg


@func_wrapper
def assert_is_none(a, msg=''):
    """Raise assertion error if a is not None."""
    assert a is None, msg


@func_wrapper
def assert_is_not_none(a, msg=''):
    """Raise assertion error if a is None."""
    assert a is not None, msg


@func_wrapper
def assert_in(a, b, msg=''):
    """Raise assertion error if a in b."""
    assert a in b, msg


@func_wrapper
def assert_not_in(a, b, msg=''):
    """Raise assertion error if a in b."""
    assert a not in b, msg


@func_wrapper
def assert_is_instance(a, b, msg=''):
    """Raise assertion error if a is instance of b."""
    assert isinstance(a, b), msg


@func_wrapper
def assert_is_not_instance(a, b, msg=''):
    """Raise assertion error if a is not instance of b."""
    assert not isinstance(a, b), msg


@func_wrapper
def assert_greater(a, b, msg=""):
    """Raise assertion error if a is not greater then b."""
    assert a > b, msg


@func_wrapper
def assert_greater_equal(a, b, msg=""):
    """Raise assertion error if a is not greater or equal then b."""
    assert a >= b, msg


@func_wrapper
def assert_less(a, b, msg=""):
    """Raise assertion error if a is not less b."""
    assert a < b, msg


@func_wrapper
def assert_less_equal(a, b, msg=""):
    """Raise assertion error if a is not less or equal then b."""
    assert a <= b, msg
