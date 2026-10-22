.. include:: ../references.txt

Testing helpers
===============

:mod:`astropop.testing` provides assertion functions used by the library's test
suite. A failed condition raises ``AssertionError`` immediately; these helpers
do not accumulate failures or continue a failed test.

Values, arrays, and identity
----------------------------

``assert_equal`` and ``assert_almost_equal`` support scalar and array inputs.
``assert_almost_equal`` uses decimal places, rather than a relative tolerance.
For numerical algorithms whose scale matters, NumPy's
``assert_allclose(actual, expected, rtol=..., atol=...)`` may express the
intended tolerance more clearly. Call ``assert_equal(actual, expected)`` for
exact equality, ``assert_almost_equal(actual, expected, decimal=6)`` for six
places, or ``assert_true(condition)`` for a Boolean condition in a test.
Successful checks return without printing; failures raise ``AssertionError``.

The remaining helpers make the condition explicit in their names. The call
forms below describe their use in test code; successful calls return ``None``
and failed conditions raise ``AssertionError``. ``msg`` replaces the default
failure explanation.

.. list-table:: Conditions checked by each helper
    :header-rows: 1
    :widths: 48 52

    * - Call form
      - Condition required for success
    * - ``assert_not_equal(a, b, msg=None)``
      - Values differ; NumPy equality semantics apply to arrays.
    * - ``assert_true(a, msg=None)`` / ``assert_false(a, msg=None)``
      - ``a`` is truthy / falsy; these do not require the singleton True / False.
    * - ``assert_is(a, b, msg=None)`` / ``assert_is_not(a, b, msg=None)``
      - Objects have the same / different identity, independently of equality.
    * - ``assert_is_none(a, msg=None)`` / ``assert_is_not_none(a, msg=None)``
      - ``a`` is / is not the singleton None.
    * - ``assert_in(a, b, msg=None)`` / ``assert_not_in(a, b, msg=None)``
      - Container ``b`` contains / does not contain ``a``.
    * - ``assert_is_instance(a, b, msg=None)`` / ``assert_is_not_instance(a, b, msg=None)``
      - ``isinstance(a, b)`` is true / false, including subclass instances.
    * - ``assert_greater(a, b, msg=None)`` / ``assert_greater_equal(a, b, msg=None)``
      - ``a > b`` / ``a >= b``.
    * - ``assert_less(a, b, msg=None)`` / ``assert_less_equal(a, b, msg=None)``
      - ``a < b`` / ``a <= b``.

These conditions do not mask errors in the comparison itself. For example,
a multi-element Boolean array is not a scalar truth value; use an array-aware
comparison instead of passing it to ``assert_true``.

Exceptions, warnings, and paths
-------------------------------

``assert_raises``, ``assert_raises_regex``, ``assert_warns``, and
``assert_not_warnings`` wrap NumPy's corresponding testing context managers.
Pass an explicit expected exception or warning class to the context manager
and put the operation being checked inside its block. Unexpected exceptions
propagate instead of being treated as a successful check.

``assert_path_exists`` and ``assert_path_not_exists`` check filesystem paths.
Create disposable fixtures in a test-specific temporary directory rather than
in the source tree. See :doc:`testing` for running the project's offline
tests, documentation checks, and separate live catalog conformance.

Recent NumPy versions deprecate the upstream warning helpers. For new code,
prefer ``pytest.warns`` in tests or ``warnings.catch_warnings`` when inspecting
warnings interactively. The latter lets you display the warning category and
message directly::

    >>> import warnings

    >>> with warnings.catch_warnings(record=True) as recorded:
    ...
    ...     # Record even repeated warnings rather than relying on the process warning filter.
    ...     warnings.simplefilter('always')
    ...     warnings.warn('example warning', UserWarning)

    >>> for item in recorded:
    ...     print(f'{item.category.__name__}: {item.message}')
    UserWarning: example warning




Reference/API
-------------

.. automodapi:: astropop.testing
    :skip: assert_warns, assert_not_warnings
    :no-inheritance-diagram:

.. currentmodule:: astropop.testing

.. py:function:: assert_warns(warning_class, *args, **kwargs)

    Alias of :func:`numpy.testing.assert_warns`. Use it as a context manager to
    require a warning, or pass a callable and its arguments to invoke it under
    the warning check. The assertion fails if the requested warning does not
    occur.

.. py:function:: assert_not_warnings(*args, **kwargs)

    Alias of :func:`numpy.testing.assert_no_warnings`. Use it as a context
    manager or pass a callable and arguments to require warning-free execution.


.. currentmodule:: None
