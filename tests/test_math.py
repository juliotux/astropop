# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
from astropop.math.hasher import hasher
from astropop.math.array import xy2r, iraf_indices, trim_array,  \
                                all_equal
import numpy as np

from astropop.testing import *


def test_hasher():
    s = 'asdf1234 &*()[]'
    h = hasher(s, 10)
    assert_equal(h, '4b37febb5e')


def test_xy2r():
    f = np.arange(4).reshape((2, 2))
    x, y = iraf_indices(f)
    r, outf = xy2r(x, y, f, 0.0, 0.0)
    assert_equal(r, [0, 1, 1, np.sqrt(2)])
    assert_equal(f.ravel(), outf)  # no reordering expected


def test_iraf_indices():
    f = np.arange(4).reshape((2, 2))
    x, y = iraf_indices(f)
    assert_equal(x, [[0, 1], [0, 1]])
    assert_equal(y, [[0, 0], [1, 1]])


class Test_AllEqual():
    def test_all_equal_1d_true(self):
        a = np.ones(10)
        assert_true(all_equal(a))

        a = np.zeros(10)
        assert_true(all_equal(a))

        a = np.array(['aAa']*10)
        assert_true(all_equal(a))

    def test_all_equal_1d_false(self):
        a = np.ones(10)
        a[2] = 2
        assert_false(all_equal(a))

        a = np.zeros(10)
        a[2] = 2
        assert_false(all_equal(a))

        a = np.array(['aAa']*10)
        a[2] = 'bBb'
        assert_false(all_equal(a))

    def test_all_equal_2d_true(self):
        a = np.ones((5, 5))
        assert_true(all_equal(a))

        a = np.zeros((5, 5))
        assert_true(all_equal(a))

        a = ['aAa']*5
        a = np.array([a]*5)
        assert_true(all_equal(a))

    def test_all_equal_2d_false(self):
        a = np.ones((5, 5))
        a[2][2] = 2
        assert_false(all_equal(a))

        a = np.zeros((5, 5))
        a[2][2] = 2
        assert_false(all_equal(a))

        a = ['aAa']*5
        a = np.array([a]*5)
        a[2][2] = 'bBb'
        assert_false(all_equal(a))


class Test_TrimArray():
    def test_trim_array_centered(self):
        y, x = np.indices((100, 100))
        a = x*y
        ta, tx, ty = trim_array(a, 21, (50, 50), (y, x))
        assert_equal(ta, a[39:61, 39:61])
        assert_equal(tx, x[39:61, 39:61])
        assert_equal(ty, y[39:61, 39:61])
        assert_equal(np.min(tx), 39)
        assert_equal(np.max(tx), 60)
        assert_equal(np.min(ty), 39)
        assert_equal(np.max(ty), 60)
        assert_equal(np.min(ta), 39*39)
        assert_equal(np.max(ta), 60*60)

    def test_trim_array_right(self):
        y, x = np.indices((100, 100))
        a = x*y
        ta, tx, ty = trim_array(a, 21, (95, 50), (y, x))
        assert_equal(ta, a[39:61, 84:])
        assert_equal(tx, x[39:61, 84:])
        assert_equal(ty, y[39:61, 84:])
        assert_equal(np.min(tx), 84)
        assert_equal(np.max(tx), 99)
        assert_equal(np.min(ty), 39)
        assert_equal(np.max(ty), 60)
        assert_equal(np.min(ta), 84*39)
        assert_equal(np.max(ta), 99*60)

    def test_trim_array_left(self):
        y, x = np.indices((100, 100))
        a = x*y
        ta, tx, ty = trim_array(a, 21, (5, 50), (y, x))
        assert_equal(ta, a[39:61, :16])
        assert_equal(tx, x[39:61, :16])
        assert_equal(ty, y[39:61, :16])
        assert_equal(np.min(tx), 0)
        assert_equal(np.max(tx), 15)
        assert_equal(np.min(ty), 39)
        assert_equal(np.max(ty), 60)
        assert_equal(np.min(ta), 0*39)
        assert_equal(np.max(ta), 15*60)

    def test_trim_array_bottom(self):
        y, x = np.indices((100, 100))
        a = x*y
        ta, tx, ty = trim_array(a, 21, (50, 5), (y, x))
        assert_equal(ta, a[:16, 39:61])
        assert_equal(tx, x[:16, 39:61])
        assert_equal(ty, y[:16, 39:61])
        assert_equal(np.min(ty), 0)
        assert_equal(np.max(ty), 15)
        assert_equal(np.min(tx), 39)
        assert_equal(np.max(tx), 60)
        assert_equal(np.min(ta), 0*39)
        assert_equal(np.max(ta), 15*60)

    def test_trim_array_top(self):
        y, x = np.indices((100, 100))
        a = x*y
        ta, tx, ty = trim_array(a, 21, (50, 95), (y, x))
        assert_equal(ta, a[84:, 39:61])
        assert_equal(tx, x[84:, 39:61])
        assert_equal(ty, y[84:, 39:61])
        assert_equal(np.min(ty), 84)
        assert_equal(np.max(ty), 99)
        assert_equal(np.min(tx), 39)
        assert_equal(np.max(tx), 60)
        assert_equal(np.min(ta), 84*39)
        assert_equal(np.max(ta), 99*60)

    def test_trim_array_no_indices(self):
        y, x = np.indices((100, 100))
        a = x*y
        ta, tx, ty = trim_array(a, 21, (50, 95))
        assert_equal(ta, a[84:, 39:61])
        assert_equal(tx, 11)
        assert_equal(ty, 11)

    def test_trim_array_one_origin(self):
        y, x = np.indices((100, 100))
        a = x*y
        ta, tx, ty = trim_array(a, 21, (50, 95), origin=1)
        assert_equal(ta, a[85:, 40:62])
        assert_equal(tx, 11)
        assert_equal(ty, 11)

    def test_trim_array_even(self):
        y, x = np.indices((100, 100))
        a = x*y
        ta, tx, ty = trim_array(a, 20, (50, 95))
        assert_equal(ta, a[85:, 40:61])
        assert_equal(tx, 10)
        assert_equal(ty, 10)
