# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import os
import mmap
import pytest
from astropop.framedata import MemMapArray, EmptyDataError
from astropop.framedata.memmap import create_array_memmap, \
                                      delete_array_memmap
import numpy as np

from astropop.testing import *


def test_create_and_delete_memmap(tmpdir):
    # Creation
    f = os.path.join(tmpdir, 'testarray.npy')
    g = os.path.join(tmpdir, 'test2array.npy')
    a = np.ones((30, 30), dtype='f8')
    b = create_array_memmap(f, a)
    c = create_array_memmap(g, a, dtype=bool)
    assert_is_instance(b, np.memmap)
    assert_is_instance(c, np.memmap)
    assert_equal(a, b)
    assert_almost_equal(a, c)
    assert_true(os.path.exists(f))
    assert_true(os.path.exists(g))

    # Deletion
    # Since for the uses the object is overwritten, we do it here too
    d = delete_array_memmap(b, read=True, remove=False)
    e = delete_array_memmap(c, read=True, remove=False)
    assert_is_not_instance(d, np.memmap)
    assert_is_not_instance(e, np.memmap)
    assert_is_instance(d, np.ndarray)
    assert_is_instance(e, np.ndarray)
    assert_equal(a, d)
    assert_almost_equal(a, e)
    assert_true(os.path.exists(f))
    assert_true(os.path.exists(g))

    d = delete_array_memmap(b, read=False, remove=True)
    e = delete_array_memmap(c, read=False, remove=True)
    assert_true(d is None)
    assert_true(e is None)
    assert_false(os.path.exists(f))
    assert_false(os.path.exists(g))

    # None should not raise errors
    create_array_memmap('dummy', None)
    delete_array_memmap(None)


class Test_MemMapArray:
    @pytest.mark.parametrize('memmap', [True, False])
    def test_create_empty_memmap(self, tmpdir, memmap):
        f = os.path.join(tmpdir, 'empty.npy')
        a = MemMapArray(None, filename=f, dtype=None, memmap=memmap)
        assert_equal(a.filename, f)
        assert_true(a._contained is None)
        assert_equal(a.memmap, memmap)
        assert_true(a.empty)
        assert_false(os.path.exists(f))
        with pytest.raises(EmptyDataError):
            # dtype whould rise
            a.dtype
        with pytest.raises(EmptyDataError):
            # shape whould rise
            a.shape
        with pytest.raises(EmptyDataError):
            # item whould rise
            a[0]
        with pytest.raises(EmptyDataError):
            # set item whould rise
            a[0] = 1

    @pytest.mark.parametrize('memmap', [True, False])
    def test_create_memmap(self, tmpdir, memmap):
        f = os.path.join(tmpdir, 'npn_empty.npy')
        arr = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
        a = MemMapArray(arr, filename=f, dtype=None, memmap=memmap)
        assert_equal(a.filename, f)
        assert_equal(a, arr)
        assert_false(a.empty)
        assert_equal(a.memmap, memmap)
        assert_equal(os.path.exists(f), memmap)
        assert_true(a.dtype == np.int64)

        a[0][0] = 10
        assert_equal(a[0][0], 10)

        a[0][:] = 20
        assert_equal(a[0], [20, 20, 20, 20, 20, 20])


    def test_enable_disable_memmap(self, tmpdir):
        f = os.path.join(tmpdir, 'npn_empty.npy')
        arr = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
        a = MemMapArray(arr, filename=f, dtype=None, memmap=False)
        assert_false(a.memmap)
        assert_false(os.path.exists(f))

        a.enable_memmap()
        assert_true(a.memmap)
        assert_true(os.path.exists(f))
        assert_is_instance(a._contained, np.memmap)

        # First keep the file
        a.disable_memmap(remove=False)
        assert_false(a.memmap)
        assert_true(os.path.exists(f))
        assert_is_not_instance(a._contained, np.memmap)

        a.enable_memmap()
        assert_true(a.memmap)
        assert_true(os.path.exists(f))
        assert_is_instance(a._contained, np.memmap)

        # Remove the file
        a.disable_memmap(remove=True)
        assert_false(a.memmap)
        assert_false(os.path.exists(f))
        assert_is_not_instance(a._contained, np.memmap)

        with pytest.raises(ValueError):
            # raises error if name is locked
            a.enable_memmap('not_the_same_name.npy')

    def test_reset_data(self, tmpdir):
        d1 = np.array([[1, 2], [3, 4]]).astype('float32')
        m = MemMapArray(d1, os.path.join(tmpdir, 'reset.npy'),
                        dtype='float64', memmap=True)
        assert_true(np.issubdtype(m.dtype, np.float64))
        assert_equal(m, d1)
        assert_false(m.empty)
        assert_true(m.memmap)

        m.reset_data(d1)
        assert_true(np.issubdtype(m.dtype, np.float32))
        assert_equal(m, d1)
        assert_false(m.empty)
        assert_true(m.memmap)

        m.reset_data(d1.astype('int16'))
        assert_true(np.issubdtype(m.dtype, np.int16))
        assert_equal(m, d1)
        assert_false(m.empty)
        assert_true(m.memmap)

        m.reset_data(None)
        assert_true(m.empty)
        assert_true(m._contained is None)
        assert_true(m.memmap)
        m.disable_memmap()

        m.reset_data(np.ones((10, 10)), dtype='float32')
        assert_true(np.issubdtype(m.dtype, np.float32))
        assert_equal(m, np.ones((10, 10)))
        assert_false(m.empty)
        assert_false(m.memmap)


# TODO: flush
# TODO: repr

###############################################################################
# For math tests we supose that numpy's math is correct
###############################################################################

parametrize_matrice = pytest.mark.parametrize('memmap, value, other',
                                              [(True, 3, 2), (False, 3, 2),
                                               (True, 1.5, 3.5),
                                               (False, 1.5, 3.5),
                                               (True, 4, 0), (False, 4, 0),
                                               (True, 0, 4), (False, 0, 4),
                                               (True, 1.5, -2),
                                               (False, 1.5, -2),
                                               (True, -10, 3.5),
                                               (False, -10, 3.5),
                                               (True, 10, 3.5),
                                               (False, 10, 3.5),
                                               (True, 1, np.nan),
                                               (False, 1, np.nan),
                                               (True, 1, np.inf),
                                               (False, 1, np.inf)])


@parametrize_matrice
def test_memmap_lt(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'lt.npy')
    arr = np.arange(0, 10, 1)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr < other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a < other
    else:
        ap_v = a < other
        assert_equal(ap_v, np_v)


@parametrize_matrice
def test_memmap_le(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'le.npy')
    arr = np.arange(0, 10, 1)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr <= other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a <= other
    else:
        ap_v = a <= other
        assert_equal(ap_v, np_v)


@parametrize_matrice
def test_memmap_gt(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'gt.npy')
    arr = np.arange(0, 10, 1)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr > other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a > other
    else:
        ap_v = a > other
        assert_equal(ap_v, np_v)


@parametrize_matrice
def test_memmap_ge(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'ge.npy')
    arr = np.arange(0, 10, 1)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr >= other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a >= other
    else:
        ap_v = a >= other
        assert_equal(ap_v, np_v)


@parametrize_matrice
def test_memmap_eq(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'eq.npy')
    arr = np.arange(0, 10, 1)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr == other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a == other
    else:
        ap_v = a == other
        assert_equal(ap_v, np_v)


@parametrize_matrice
def test_memmap_ne(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'ne.npy')
    arr = np.arange(0, 10, 1)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr != other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a != other
    else:
        ap_v = a != other
        assert_equal(ap_v, np_v)


@parametrize_matrice
def test_math_add(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'add.npy')
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr+other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a+other
    else:
        ap_v = a+other
        assert_equal(ap_v, np_v)

    try:
        arr += other
    except Exception as e:
        with pytest.raises(e.__class__):
            a += other
    else:
        a += other
        assert_equal(a, arr)


@parametrize_matrice
def test_math_sub(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'sun.npy')
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr-other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a-other
    else:
        ap_v = a-other
        assert_equal(ap_v, np_v)

    try:
        arr -= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a -= other
    else:
        a -= other
        assert_equal(a, arr)


@parametrize_matrice
def test_math_pow(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'pow.npy')
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr**other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a**other
    else:
        ap_v = a**other
        assert_equal(ap_v, np_v)

    try:
        arr **= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a **= other
    else:
        a **= other
        assert_equal(a, arr)


@parametrize_matrice
def test_math_truediv(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'div.npy')
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr/other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a/other
    else:
        ap_v = a/other
        assert_equal(ap_v, np_v)

    try:
        arr /= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a /= other
    else:
        a /= other
        assert_equal(a, arr)


@parametrize_matrice
def test_math_floordiv(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'floordiv.npy')
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr//other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a//other
    else:
        ap_v = a//other
        assert_equal(ap_v, np_v)

    try:
        arr //= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a //= other
    else:
        a //= other
        assert_equal(a, arr)


@parametrize_matrice
def test_math_mul(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'mul.npy')
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr*other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a*other
    else:
        ap_v = a*other
        assert_equal(ap_v, np_v)

    try:
        arr *= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a *= other
    else:
        a *= other
        assert_equal(a, arr)


@parametrize_matrice
def test_math_mod(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'mod.npy')
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr % other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a % other
    else:
        ap_v = a % other
        assert_equal(ap_v, np_v)

    try:
        arr %= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a %= other
    else:
        a %= other
        assert_equal(a, arr)


@parametrize_matrice
def test_math_lshift(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'lshift.npy')
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr << other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a << other
    else:
        ap_v = a << other
        assert_equal(ap_v, np_v)

    try:
        arr <<= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a <<= other
    else:
        a <<= other
        assert_equal(a, arr)
    # FIXME: why do this simply don't work like in numpy array???
    # assert_equal(other<<arr, other<<a)


@parametrize_matrice
def test_math_rshift(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'rshift.npy')
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr >> other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a >> other
    else:
        ap_v = a >> other
        assert_equal(ap_v, np_v)

    try:
        arr >>= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a >>= other
    else:
        a >>= other
        assert_equal(a, arr)
    # FIXME: why do this simply don't work like in numpy array???
    # assert_equal(other>>arr, other>>a)


@parametrize_matrice
def test_math_and(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'and.npy')
    arr = np.array([[0, 1, 0, 1], [1, 0, 1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr & other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a & other
    else:
        ap_v = a & other
        assert_equal(ap_v, np_v)

    try:
        arr &= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a &= other
    else:
        a &= other
        assert_equal(a, arr)
    # FIXME: why do this simply don't work like in numpy array???
    # assert_equal(other & arr, other & a)


@parametrize_matrice
def test_math_or(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'xor.npy')
    arr = np.array([[0, 1, 0, 1], [1, 0, 1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr | other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a | other
    else:
        ap_v = a | other
        assert_equal(ap_v, np_v)

    try:
        arr |= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a |= other
    else:
        a |= other
        assert_equal(a, arr)
    # FIXME: why do this simply don't work like in numpy array???
    # assert_equal(other | arr, other | a)


@parametrize_matrice
def test_math_xor(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'xor.npy')
    arr = np.array([[0, 1, 0, 1], [1, 0, 1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr ^ other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a ^ other
    else:
        ap_v = a ^ other
        assert_equal(ap_v, np_v)

    try:
        arr ^= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a ^= other
    else:
        a ^= other
        assert_equal(a, arr)
    # FIXME: why do this simply don't work like in numpy array???
    # assert_equal(other | arr, other | a)


@parametrize_matrice
def test_math_neg(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'neg.npy')
    arr = np.array([[0, 1, 0, -1], [1, 0, -1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = -arr
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = -a
    else:
        ap_v = -a
        assert_equal(ap_v, np_v)


@parametrize_matrice
def test_math_pos(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'pos.npy')
    arr = np.array([[0, 1, 0, -1], [1, 0, -1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = +arr
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = +a
    else:
        ap_v = +a
        assert_equal(ap_v, np_v)


@parametrize_matrice
def test_math_abs(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'abs.npy')
    arr = np.array([[0, 1, 0, -1], [1, 0, -1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr.__abs__()
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a.__abs__()
    else:
        ap_v = a.__abs__()
        assert_equal(ap_v, np_v)


@parametrize_matrice
def test_math_invert(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'invert.npy')
    arr = np.array([[0, 1, 0, -1], [1, 0, -1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = ~arr
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = ~a
    else:
        ap_v = ~a
        assert_equal(ap_v, np_v)


@parametrize_matrice
def test_math_matmul(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'invert.npy')
    arr = np.array([[0, 1, 0, -1], [1, 0, -1, 0]]) * value
    other = np.random.randn(4, 2)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    try:
        np_v = arr@other
    except Exception as e:
        with pytest.raises(e.__class__):
            ap_v = a@other
    else:
        ap_v = a@other
        assert_equal(ap_v, np_v)


@parametrize_matrice
def test_math_bool_all_any(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'bool.npy')
    arr = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
    a = MemMapArray(arr, filename=f, memmap=memmap)
    with pytest.raises(ValueError):
        bool(a)
    assert_false(a.all())
    assert_true(a.any())

    arr = np.array([0, 0, 0])
    a = MemMapArray(arr, filename=f+'1', memmap=memmap)
    assert_false(a.all())
    assert_false(a.any())

    arr = np.array([1, 1, 1])
    a = MemMapArray(arr, filename=f+'2', memmap=memmap)
    assert_true(a.all())
    assert_true(a.any())


@parametrize_matrice
def test_math_float(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'float.npy')
    arr = np.arange(10, dtype='int8')*value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    with pytest.raises(TypeError):
        float(a)

    a = MemMapArray([value], filename=f, memmap=memmap)
    assert_equal(float(value), float(a))


@parametrize_matrice
def test_math_int(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'int.npy')
    arr = np.arange(10, dtype='int8')*value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    with pytest.raises(TypeError):
        int(a)

    a = MemMapArray([value], filename=f, memmap=memmap)
    assert_equal(int(value), int(a))


@parametrize_matrice
def test_math_complex(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'complex.npy')
    arr = np.arange(10, dtype='int8')*value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    with pytest.raises(TypeError):
        complex(a)

    a = MemMapArray([value], filename=f, memmap=memmap)
    try:
        complex(np.array([value]), other)
    except Exception as e:
        with pytest.raises(e.__class__):
            complex(a, other)
    else:
        try:
            assert_equal(complex(value), complex(arr))
            assert_equal(complex(value, other), complex(arr, other))
        except Exception as e:
            with pytest.raises(e.__class__):
                assert_equal(complex(value), complex(arr))
                assert_equal(complex(value, other), complex(arr, other))
        else:
            assert_equal(complex(value), complex(arr))
            assert_equal(complex(value, other), complex(arr, other))


@pytest.mark.parametrize('memmap', [True, False])
def test_math_len(tmpdir, memmap):
    f = os.path.join(tmpdir, 'len.npy')
    for i in [np.arange(10), np.array([1]), np.zeros((10, 10)),
              np.zeros((10, 10, 10)), np.array(None)]:
        arr = i
        a = MemMapArray(arr, filename=f, memmap=memmap)
        try:
            np_v = len(arr)
        except Exception as e:
            with pytest.raises(e.__class__):
                ap_v = len(a)
        else:
            ap_v = len(a)
            assert_equal(np_v, ap_v)


@pytest.mark.parametrize('memmap', [True, False])
def test_math_redirects(tmpdir, memmap):
    f = os.path.join(tmpdir, 'redirects.npy')

    def check_arr(arr, a):
        arr_flags = arr.flags
        a_flags = a.flags
        for i in ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'WRITEABLE', 'ALIGNED',
                  'WRITEBACKIFCOPY', 'FNC', 'FORC',
                  'BEHAVED', 'CARRAY', 'FARRAY']:
            assert_equal(arr_flags[i], a_flags[i])
        for i in ['OWNDATA']:
            if memmap:
                assert_not_equal(arr_flags[i], a_flags[i])
            else:
                assert_equal(arr_flags[i], a_flags[i])

        if memmap:
            assert_is_instance(a.base, mmap.mmap)
        else:
            assert_true(a.base is None)

        assert_equal(arr.shape, a.shape)
        assert_equal(arr.strides, a.strides)
        assert_equal(arr.ndim, a.ndim)
        assert_equal(arr.data, a.data)
        assert_equal(arr.size, a.size)
        assert_equal(arr.itemsize, a.itemsize)
        assert_equal(arr.nbytes, a.nbytes)
        assert_equal(arr.dtype, a.dtype)

        assert_is_instance(a.tolist(), list)
        assert_equal(arr.tolist(), a.tolist())
        assert_is_instance(a.tostring(), bytes)
        assert_equal(arr.tostring(), a.tostring())
        assert_is_instance(a.tobytes(), bytes)
        assert_equal(arr.tobytes(), a.tobytes())
        assert_is_instance(a.dumps(), bytes)
        # FIXME: assert_equal(arr.dumps(), a.dumps())

        assert_equal(arr.T, a.T)
        assert_equal(arr.transpose(), a.transpose())
        assert_equal(arr.flatten(), a.flatten())
        assert_equal(arr.ravel(), a.ravel())
        assert_equal(arr.squeeze(), a.squeeze())
        assert_equal(arr.argsort(), a.argsort())
        assert_equal(arr.argpartition(1), a.argpartition(1))
        assert_equal(arr.nonzero(), a.nonzero())
        assert_equal(arr.max(), a.max())
        assert_equal(arr.argmax(), a.argmax())
        assert_equal(arr.min(), a.min())
        assert_equal(arr.argmin(), a.argmin())
        assert_equal(arr.max(axis=0), a.max(axis=0))
        assert_equal(arr.min(axis=0), a.min(axis=0))
        assert_equal(arr.argmax(axis=0), a.argmax(axis=0))
        assert_equal(arr.argmin(axis=0), a.argmin(axis=0))
        assert_equal(arr.real, a.real)
        assert_equal(arr.imag, a.imag)
        assert_equal(arr.round(), a.round())
        assert_equal(arr.sum(), a.sum())
        assert_equal(arr.sum(axis=0), a.sum(axis=0))
        assert_equal(arr.cumsum(), a.cumsum())
        assert_equal(arr.cumsum(axis=0), a.cumsum(axis=0))
        assert_equal(arr.mean(), a.mean())
        assert_equal(arr.mean(axis=0), a.mean(axis=0))
        assert_equal(arr.var(), a.var())
        assert_equal(arr.var(axis=0), a.var(axis=0))
        assert_equal(arr.std(), a.std())
        assert_equal(arr.std(axis=0), a.std(axis=0))
        assert_equal(arr.prod(), a.prod())
        assert_equal(arr.prod(axis=0), a.prod(axis=0))
        assert_equal(arr.cumprod(), a.cumprod())
        assert_equal(arr.cumprod(axis=0), a.cumprod(axis=0))

        for i, j in zip(arr.flat, a.flat):
            assert_equal(i, j)
        for i in range(9):
            assert_equal(arr.item(i), a.item(i))

        assert_equal(arr.astype(bool), a.astype(bool))
        assert_equal(arr.astype(int), a.astype(int))

        assert_equal(arr.all(), a.all())
        assert_equal(arr.any(), a.any())

        # FIXME: assert_equal(arr.ctypes, a.ctypes)

        # TODO: itemset
        # TODO: tofile
        # TODO: dump
        # TODO: byteswap
        # TODO: copy
        # TODO: view
        # TODO: setflags
        # TODO: getfield
        # TODO: reshape
        # TODO: resize
        # TODO: take
        # TODO: put
        # TODO: repeat
        # TODO: sort
        # TODO: choose
        # TODO: partition
        # TODO: searchsorted
        # TODO: compress
        # TODO: ptp
        # TODO: conj
        # TODO: swapaxes
        # TODO: diagonal
        # TODO: trace

    x = np.random.randint(9, size=(3, 3)).astype(np.float)
    y = MemMapArray(x, filename=f, memmap=memmap)
    check_arr(x, y)

    x = np.zeros((5, 5)).astype(np.float)
    y = MemMapArray(x, filename=f, memmap=memmap)
    check_arr(x, y)

    x = np.ones((5, 5)).astype(np.float)
    y = MemMapArray(x, filename=f, memmap=memmap)
    check_arr(x, y)

    x = np.arange(20).astype(np.float)
    y = MemMapArray(x, filename=f, memmap=memmap)
    check_arr(x, y)
