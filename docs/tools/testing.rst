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

``assert_is``/``assert_is_not`` check object identity. The None, membership,
instance, and ordering helpers check their named conditions. Most accept a
custom ``msg`` to explain the failure.

Exceptions, warnings, and paths
-------------------------------

``assert_raises``, ``assert_raises_regex``, ``assert_warns``, and
``assert_not_warnings`` wrap NumPy's corresponding testing context managers.
Pass an explicit expected exception or warning class to the context manager
and put the operation being checked inside its block. Unexpected exceptions
propagate instead of being treated as a successful check.

``assert_path_exists`` and ``assert_path_not_exists`` check filesystem paths.
Create disposable fixtures in a test-specific temporary directory rather than
in the source tree. See :doc:`../dev/testing` for running the project's offline
tests, documentation checks, and separate live catalog conformance.

Recent NumPy versions deprecate the upstream warning helpers. For new code,
prefer ``pytest.warns`` in tests or ``warnings.catch_warnings`` when inspecting
warnings interactively. The latter lets you display the warning category and
message directly::

    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as recorded:
    ...     warnings.simplefilter('always')
    ...     warnings.warn('example warning', UserWarning)
    >>> for item in recorded:
    ...     print(f'{item.category.__name__}: {item.message}')
    UserWarning: example warning

Testing helpers API
-------------------

See :doc:`../reference/utilities` for the API reference.

.. raw:: html

    <span id="astropop-testing-module"></span>
    <span id="astropop.testing.assert_not_warnings"></span>
    <span id="astropop.testing.assert_warns"></span>
    <span id="functions"></span>
    <span id="module-astropop.testing"></span>
    <span id="notes"></span>
