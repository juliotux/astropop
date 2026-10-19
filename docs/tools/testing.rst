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
``np.testing.assert_allclose`` may express the intended tolerance more clearly::

    >>> from astropop.testing import assert_equal, assert_almost_equal, assert_true
    >>> assert_equal([1, 2], [1, 2])
    >>> assert_almost_equal(1.0, 1.0000001, decimal=6)
    >>> assert_true(2 > 1)

``assert_is``/``assert_is_not`` check object identity. The None, membership,
instance, and ordering helpers check their named conditions. Most accept a
custom ``msg`` to explain the failure.

Exceptions, warnings, and paths
-------------------------------

``assert_raises``, ``assert_raises_regex``, ``assert_warns``, and
``assert_not_warnings`` wrap NumPy's corresponding testing context managers.
Use explicit expected exception or warning classes::

    >>> from astropop.testing import assert_raises
    >>> with assert_raises(ValueError):
    ...     int('not a number')

``assert_path_exists`` and ``assert_path_not_exists`` check filesystem paths.
Create disposable fixtures in a test-specific temporary directory rather than
in the source tree. See :doc:`../dev/testing` for running the project's offline
tests, documentation checks, and separate live catalog conformance.

Testing helpers API
-------------------

.. automodapi:: astropop.testing
    :no-inheritance-diagram:
