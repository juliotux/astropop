# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import tempfile
from astropop.testing import (assert_equal, assert_not_equal,
                              assert_almost_equal,
                              assert_true, assert_false,
                              assert_is, assert_is_not,
                              assert_in, assert_not_in,
                              assert_is_none, assert_is_not_none,
                              assert_greater, assert_greater_equal,
                              assert_less, assert_less_equal,
                              assert_is_instance, assert_is_not_instance,
                              assert_path_exists, assert_path_not_exists)
import numpy as np
import pytest


class TestTestingHelpers():
    """Test the testing helpers."""

    def test_numpy_err_msg(self):
        # Ensure compatibility with numpy testing
        with pytest.raises(AssertionError):
            assert_almost_equal([1, 2], [3, 4], 0, None)
        with pytest.raises(AssertionError):
            assert_equal([1, 2], [3, 4], None)
        with pytest.raises(AssertionError):
            assert_almost_equal([1, 2], [3, 4], 0, '')
        with pytest.raises(AssertionError):
            assert_equal([1, 2], [3, 4], '')

    def test_assert_equal(self):
        assert_equal(1, 1)
        assert_equal(np.arange(4), [0, 1, 2, 3])
        assert_equal([np.nan, np.inf, 0, 1], [np.nan, np.inf, 0, 1])

        with pytest.raises(AssertionError):
            assert_equal(1, 2)
        with pytest.raises(AssertionError):
            assert_equal(np.arange(5), [0, 1, 2, 3])
        with pytest.raises(AssertionError):
            assert_equal([np.nan, 0, 1], [0, 0, 1])

    def test_assert_not_equal(self):
        assert_not_equal(1, 2)
        assert_not_equal(np.arange(5), [0, 1, 2, 3])
        assert_not_equal([np.nan, 0, 1], [0, 0, 1])

        with pytest.raises(AssertionError):
            assert_not_equal(1, 1)
        with pytest.raises(AssertionError):
            assert_not_equal(np.arange(4), [0, 1, 2, 3])
        with pytest.raises(AssertionError):
            assert_not_equal([np.nan, np.inf, 0, 1], [np.nan, np.inf, 0, 1])

    def test_assert_almost_equal(self):
        assert_almost_equal(1.0578602, 1.0578603)
        with pytest.raises(AssertionError):
            assert_almost_equal(1, 2)

    def test_assert_true(self):
        assert_true(True)
        with pytest.raises(AssertionError):
            assert_true(False)

    def test_assert_false(self):
        assert_false(False)
        with pytest.raises(AssertionError):
            assert_false(True)

    def test_assert_is(self):
        a = [1]
        b = a
        assert_is(a, b)
        with pytest.raises(AssertionError):
            assert_is(a, [])

    def test_assert_is_not(self):
        a = [1]
        b = [1]
        assert_is_not(a, b)
        with pytest.raises(AssertionError):
            c = a
            assert_is_not(a, c)

    def test_assert_is_none(self):
        a = None
        assert_is_none(a)
        with pytest.raises(AssertionError):
            assert_is_none(1)

    def test_assert_is_not_none(self):
        a = 0
        assert_is_not_none(a)
        with pytest.raises(AssertionError):
            b = None
            assert_is_not_none(b)

    def test_assert_is_instance(self):
        a = []
        assert_is_instance(a, list)
        with pytest.raises(AssertionError):
            assert_is_instance(a, dict)

    def test_assert_is_not_instance(self):
        a = []
        assert_is_not_instance(a, dict)
        with pytest.raises(AssertionError):
            assert_is_not_instance(a, list)

    def test_assert_in(self):
        assert_in(1, [1, 2, 3, 4, 5])
        assert_in('a', 'abc')
        with pytest.raises(AssertionError):
            assert_in('d', 'abc')
        with pytest.raises(AssertionError):
            assert_in(1, [2, 3, 4])

    def test_assert_not_in(self):
        assert_not_in(6, [1, 2, 3, 4, 5])
        assert_not_in('d', 'abc')
        with pytest.raises(AssertionError):
            assert_not_in('a', 'abc')
        with pytest.raises(AssertionError):
            assert_not_in(2, [2, 3, 4])

    def test_assert_greater(self):
        assert_greater(3, 2)
        with pytest.raises(AssertionError):
            assert_greater(3, 3)
            assert_greater(2, 3)

    def test_assert_greater_equal(self):
        assert_greater_equal(3, 2)
        assert_greater_equal(3, 3)
        with pytest.raises(AssertionError):
            assert_greater_equal(2, 3)

    def test_assert_less(self):
        assert_less(1, 2)
        with pytest.raises(AssertionError):
            assert_less(3, 3)
        with pytest.raises(AssertionError):
            assert_less(4, 3)

    def test_assert_less_equal(self):
        assert_less_equal(2, 3)
        assert_less_equal(3, 3)
        with pytest.raises(AssertionError):
            assert_less_equal(4, 3)

    def test_assert_path_exists(self):
        f = tempfile.mkstemp()[1]
        assert_path_exists(f)
        with pytest.raises(AssertionError):
            assert_path_exists('this_path_do_not_exists_c1234123@'
                               '3412341233456efsccdwwefxsd')

    def test_assert_path_not_exists(self):
        f = 'this_path_do_not_exists_c1234123@3412341233456efsccdwwefxsd'
        assert_path_not_exists(f)
        with pytest.raises(AssertionError):
            f = tempfile.mkstemp()[1]
            assert_path_not_exists(f)
