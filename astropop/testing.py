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


def func_wrapper(func, docstring=None):
    """Warp the function, if needed. Now, its dummy."""
    # The idea here is to, in future, continue the tests and log all
    # failures at once. Like pytest_check.
    if docstring is not None:
        func.__doc__ = docstring
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
    if not a:
        raise AssertionError(msg)


@func_wrapper
def assert_false(a, msg=''):
    """Raise assertion error if condition is not False."""
    if a:
        raise AssertionError(msg)


assert_equal = func_wrapper(npt.assert_array_equal,
                            'Check if two objects are equal. Arrays supported.'
                            '\nImported from Numpy.')
assert_almost_equal = func_wrapper(npt.assert_array_almost_equal,
                                   'Check if two objects are almost equal. '
                                   'Arrays supported.'
                                   '\nImported from Numpy.')


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
    """Raise assertion error if `a is b`."""
    if a is not b:
        raise AssertionError(msg)


@func_wrapper
def assert_is_not(a, b, msg=''):
    """Raise assertion error if `a is not b`."""
    if a is b:
        raise AssertionError(msg)


@func_wrapper
def assert_is_none(a, msg=''):
    """Raise assertion error if `a is not None`."""
    if a is not None:
        raise AssertionError(msg)


@func_wrapper
def assert_is_not_none(a, msg=''):
    """Raise assertion error if `a is None`."""
    if a is None:
        raise AssertionError(msg)


@func_wrapper
def assert_in(a, b, msg=''):
    """Raise assertion error if `a not in b`."""
    if a not in b:
        raise AssertionError(msg)


@func_wrapper
def assert_not_in(a, b, msg=''):
    """Raise assertion error if `a in b`."""
    if a in b:
        raise AssertionError(msg)


@func_wrapper
def assert_is_instance(a, b, msg=''):
    """Raise assertion error if not `isinstance(a, b)`."""
    if not isinstance(a, b):
        raise AssertionError(msg)


@func_wrapper
def assert_is_not_instance(a, b, msg=''):
    """Raise assertion error if `isinstance(a, b)`."""
    if isinstance(a, b):
        raise AssertionError(msg)


@func_wrapper
def assert_greater(a, b, msg=""):
    """Raise assertion error if a is not greater then b."""
    if not a > b:
        raise AssertionError(msg)


@func_wrapper
def assert_greater_equal(a, b, msg=""):
    """Raise assertion error if a is not greater or equal then b."""
    if not a >= b:
        raise AssertionError(msg)


@func_wrapper
def assert_less(a, b, msg=""):
    """Raise assertion error if a is not less b."""
    if not a < b:
        raise AssertionError(msg)


@func_wrapper
def assert_less_equal(a, b, msg=""):
    """Raise assertion error if a is not less or equal then b."""
    if not a <= b:
        raise AssertionError(msg)
