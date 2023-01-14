.. include:: ../references.txt

Testing Helpers
===============

For better unit testing, we implemented (and imported), a lot of testing helpers for the most common checkings. These tests encloses some `assert`, so in a single function we can decide the best way to do this in the future.

Also, this open the doors to, *in a future release*, perform a non catastrofic failure checking, meaning that the test will continue even with failure and, in the end, all the failures are presented at once.

They are simple functions that will raise `AssertionError` if the condition they are designed is not satisfied. Like:

.. ipython:: python
    :okexcept:

    from astropop.testing import assert_true, assert_equal
    assert_true(True, 'true is true, so not raised')
    assert_true(1 == 2, 'one is not equal to two')

.. ipython:: python
    :okexcept:

    assert_equal(1, 1, 'one is equal to one, so not raised')
    assert_equal(1, 2, 'one is not equal to two')

For all the functions, check the API.

Testing Helpers API
-------------------

.. automodapi:: astropop.testing
    :no-inheritance-diagram:
