.. include:: references.txt

Testing Helpers
===============

For better unit testing, we implemented (and imported), a lot of testing helpers for the most common checkings. These tests encloses some `assert`, so in a single function we can decide the best way to do this in the future.

Also, this open the doors to, *in a future release*, perform a non catastrofic failure checking, meaning that the test will continue even with failure and, in the end, all the failures are presented at once.

They are simple functions that will raise `AssertionError` if the condition they are designed is not satisfied. Like:

    >>> from astropop.testing import assert_true, assert_equal
    >>> assert_true(True, 'true is true, so not raised')
    >>> assert_true(1 == 2, 'one is not equal to two')   # doctest: +SKIP
    AssertionError: one is not equal to two
    >>> assert_equal(1, 1, 'one is equal to one, so not raised')
    >>> assert_equal(1, 2, 'one is not equal to two')   # doctest: +SKIP
    AssertionError: one is not equal to two

For all the functions, check the API.

Testing Helpers API
-------------------

.. autofunction:: astropop.testing.assert_true
.. autofunction:: astropop.testing.assert_false
.. autofunction:: astropop.testing.assert_equal
.. autofunction:: astropop.testing.assert_not_equal
.. autofunction:: astropop.testing.assert_almost_equal
.. autofunction:: astropop.testing.assert_is
.. autofunction:: astropop.testing.assert_is_not
.. autofunction:: astropop.testing.assert_is_instance
.. autofunction:: astropop.testing.assert_is_not_instance
.. autofunction:: astropop.testing.assert_in
.. autofunction:: astropop.testing.assert_not_in
.. autofunction:: astropop.testing.assert_greater
.. autofunction:: astropop.testing.assert_greater_equal
.. autofunction:: astropop.testing.assert_less
.. autofunction:: astropop.testing.assert_less_equal
.. autofunction:: astropop.testing.assert_raises
.. autofunction:: astropop.testing.assert_raises_regex
.. autofunction:: astropop.testing.assert_warns
.. autofunction:: astropop.testing.assert_no_warnings

