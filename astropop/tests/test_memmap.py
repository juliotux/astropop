# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import mmap
import pytest
from astropop.memmap import MemMapArray, create_array_memmap, \
                            delete_array_memmap
from astropy import units as u
import numpy as np
import numpy.testing as npt


def test_create_and_delete_memmap(tmpdir):
    # Creation
    f = os.path.join(tmpdir, 'testarray.npy')
    g = os.path.join(tmpdir, 'test2array.npy')
    a = np.ones((30, 30), dtype='f8')
    b = create_array_memmap(f, a)
    c = create_array_memmap(g, a, dtype=bool)
    assert isinstance(b, np.memmap)
    assert isinstance(c, np.memmap)
    npt.assert_array_equal(a, b)
    npt.assert_allclose(a, c)
    assert os.path.exists(f)
    assert os.path.exists(g)

    # Deletion
    # Since for the uses the object is overwritten, we do it here too
    d = delete_array_memmap(b, read=True, remove=False)
    e = delete_array_memmap(c, read=True, remove=False)
    assert not isinstance(d, np.memmap)
    assert not isinstance(e, np.memmap)
    assert isinstance(d, np.ndarray)
    assert isinstance(e, np.ndarray)
    npt.assert_array_equal(a, d)
    npt.assert_allclose(a, e)
    assert os.path.exists(f)
    assert os.path.exists(g)

    d = delete_array_memmap(b, read=False, remove=True)
    e = delete_array_memmap(c, read=False, remove=True)
    assert d is None
    assert e is None
    assert not os.path.exists(f)
    assert not os.path.exists(g)

    # None should not raise errors
    create_array_memmap('dummy', None)
    delete_array_memmap(None)


@pytest.mark.parametrize('memmap', [True, False])
def test_create_empty_memmap(tmpdir, memmap):
    f = os.path.join(tmpdir, 'empty.npy')
    a = MemMapArray(None, filename=f, dtype=None, unit=None, memmap=memmap)
    assert a.filename == f
    assert a._contained is None
    assert a.memmap == memmap
    assert a.empty
    assert not os.path.exists(f)
    assert a.unit is u.dimensionless_unscaled
    with pytest.raises(KeyError):
        # dtype whould rise
        a.dtype
    with pytest.raises(KeyError):
        # shape whould rise
        a.shape
    with pytest.raises(KeyError):
        # item whould rise
        a[0]
    with pytest.raises(KeyError):
        # set item whould rise
        a[0] = 1


@pytest.mark.parametrize('memmap', [True, False])
def test_create_memmap(tmpdir, memmap):
    f = os.path.join(tmpdir, 'npn_empty.npy')
    arr = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
    a = MemMapArray(arr, filename=f, dtype=None, unit=None, memmap=memmap)
    assert a.filename == f
    npt.assert_array_equal(a, arr)
    assert not a.empty
    assert a.memmap == memmap
    assert os.path.exists(f) == memmap
    assert a.unit == u.dimensionless_unscaled
    assert a.dtype == np.int64

    a[0][0] = 10
    assert a[0][0] == 10

    a[0][:] = 20
    npt.assert_array_equal(a[0], [20, 20, 20, 20, 20, 20])


def test_enable_disable_memmap(tmpdir):
    f = os.path.join(tmpdir, 'npn_empty.npy')
    arr = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
    a = MemMapArray(arr, filename=f, dtype=None, unit=None, memmap=False)
    assert not a.memmap
    assert not os.path.exists(f)

    a.enable_memmap()
    assert a.memmap
    assert os.path.exists(f)
    assert isinstance(a._contained, np.memmap)

    # First keep the file
    a.disable_memmap(remove=False)
    assert not a.memmap
    assert os.path.exists(f)
    assert not isinstance(a._contained, np.memmap)

    a.enable_memmap()
    assert a.memmap
    assert os.path.exists(f)
    assert isinstance(a._contained, np.memmap)

    # Remove the file
    a.disable_memmap(remove=True)
    assert not a.memmap
    assert not os.path.exists(f)
    assert not isinstance(a._contained, np.memmap)

    with pytest.raises(ValueError):
        # raises error if name is locked
        a.enable_memmap('not_the_same_name.npy')


def test_units_on_creation(tmpdir):
    m = MemMapArray(None, os.path.join(tmpdir, 'unit.npy'), unit='adu')
    assert m.unit is u.adu


def test_unit_assign(tmpdir):
    m = MemMapArray(None, os.path.join(tmpdir, 'unit.npy'), unit=None)
    m.set_unit('adu')
    assert m.unit is u.adu

    with pytest.raises(AttributeError):
        # no direct assignment
        m.unit = 'adu'


def test_invalid_unit(tmpdir):
    with pytest.raises(ValueError):
        m = MemMapArray(None, unit='Invalid')

    m = MemMapArray(None)
    with pytest.raises(ValueError):
        m.set_unit('invalid')
    with pytest.raises(AttributeError):
        m.unit = 'invalid'


def test_unit_change(tmpdir):
    m = MemMapArray(None, unit='m')
    assert m.unit is u.m
    m.set_unit('adu')
    assert m.unit is u.adu


def test_reset_data(tmpdir):
    d1 = np.array([[1, 2], [3, 4]]).astype('float32')
    m = MemMapArray(d1, os.path.join(tmpdir, 'reset.npy'),
                    dtype='float64', unit='adu', memmap=True)
    assert np.issubdtype(m.dtype, np.float64)
    npt.assert_array_equal(m, d1)
    assert not m.empty
    assert m.memmap

    m.reset_data(d1)
    assert np.issubdtype(m.dtype, np.float32)
    npt.assert_array_equal(m, d1)
    assert not m.empty
    assert m.memmap

    m.reset_data(d1.astype('int16'))
    assert np.issubdtype(m.dtype, np.int16)
    npt.assert_array_equal(m, d1)
    assert not m.empty
    assert m.memmap

    m.reset_data(None)
    assert m.empty
    assert m._contained is None
    assert m.memmap
    m.disable_memmap()

    m.reset_data(np.ones((10, 10)), unit='m', dtype='float32')
    assert np.issubdtype(m.dtype, np.float32)
    npt.assert_array_equal(m, np.ones((10, 10)))
    assert not m.empty
    assert not m.memmap


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
        npt.assert_array_equal(ap_v, np_v)


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
        npt.assert_array_equal(ap_v, np_v)


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
        npt.assert_array_equal(ap_v, np_v)


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
        npt.assert_array_equal(ap_v, np_v)


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
        npt.assert_array_equal(ap_v, np_v)


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
        npt.assert_array_equal(ap_v, np_v)


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
        npt.assert_array_equal(ap_v, np_v)

    try:
        arr += other
    except Exception as e:
        with pytest.raises(e.__class__):
            a += other
    else:
        a += other
        npt.assert_array_equal(a, arr)


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
        npt.assert_array_equal(ap_v, np_v)

    try:
        arr -= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a -= other
    else:
        a -= other
        npt.assert_array_equal(a, arr)


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
        npt.assert_array_equal(ap_v, np_v)

    try:
        arr **= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a **= other
    else:
        a **= other
        npt.assert_array_equal(a, arr)


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
        npt.assert_array_equal(ap_v, np_v)

    try:
        arr /= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a /= other
    else:
        a /= other
        npt.assert_array_equal(a, arr)


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
        npt.assert_array_equal(ap_v, np_v)

    try:
        arr //= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a //= other
    else:
        a //= other
        npt.assert_array_equal(a, arr)


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
        npt.assert_array_equal(ap_v, np_v)

    try:
        arr *= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a *= other
    else:
        a *= other
        npt.assert_array_equal(a, arr)


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
        npt.assert_array_equal(ap_v, np_v)

    try:
        arr %= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a %= other
    else:
        a %= other
        npt.assert_array_equal(a, arr)


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
        npt.assert_array_equal(ap_v, np_v)

    try:
        arr <<= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a <<= other
    else:
        a <<= other
        npt.assert_array_equal(a, arr)
    # FIXME: why do this simply don't work like in numpy array???
    # npt.assert_array_equal(other<<arr, other<<a)


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
        npt.assert_array_equal(ap_v, np_v)

    try:
        arr >>= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a >>= other
    else:
        a >>= other
        npt.assert_array_equal(a, arr)
    # FIXME: why do this simply don't work like in numpy array???
    # npt.assert_array_equal(other>>arr, other>>a)


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
        npt.assert_array_equal(ap_v, np_v)

    try:
        arr &= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a &= other
    else:
        a &= other
        npt.assert_array_equal(a, arr)
    # FIXME: why do this simply don't work like in numpy array???
    # npt.assert_array_equal(other & arr, other & a)


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
        npt.assert_array_equal(ap_v, np_v)

    try:
        arr |= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a |= other
    else:
        a |= other
        npt.assert_array_equal(a, arr)
    # FIXME: why do this simply don't work like in numpy array???
    # npt.assert_array_equal(other | arr, other | a)


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
        npt.assert_array_equal(ap_v, np_v)

    try:
        arr ^= other
    except Exception as e:
        with pytest.raises(e.__class__):
            a ^= other
    else:
        a ^= other
        npt.assert_array_equal(a, arr)
    # FIXME: why do this simply don't work like in numpy array???
    # npt.assert_array_equal(other | arr, other | a)


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
        npt.assert_array_equal(ap_v, np_v)


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
        npt.assert_array_equal(ap_v, np_v)


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
        npt.assert_array_equal(ap_v, np_v)


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
        npt.assert_array_equal(ap_v, np_v)


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
        npt.assert_array_equal(ap_v, np_v)


@parametrize_matrice
def test_math_bool_all_any(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'bool.npy')
    arr = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
    a = MemMapArray(arr, filename=f, memmap=memmap)
    with pytest.raises(ValueError):
        bool(a)
    assert not a.all()
    assert a.any()

    arr = np.array([0, 0, 0])
    a = MemMapArray(arr, filename=f+'1', memmap=memmap)
    assert not a.all()
    assert not a.any()

    arr = np.array([1, 1, 1])
    a = MemMapArray(arr, filename=f+'2', memmap=memmap)
    assert a.all()
    assert a.any()


@parametrize_matrice
def test_math_float(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'float.npy')
    arr = np.arange(10, dtype='int8')*value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    with pytest.raises(TypeError):
        float(a)

    a = MemMapArray([value], filename=f, memmap=memmap)
    assert float(value) == float(a)


@parametrize_matrice
def test_math_int(tmpdir, memmap, value, other):
    f = os.path.join(tmpdir, 'int.npy')
    arr = np.arange(10, dtype='int8')*value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    with pytest.raises(TypeError):
        int(a)

    a = MemMapArray([value], filename=f, memmap=memmap)
    assert int(value) == int(a)


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
            assert complex(value) == complex(arr)
            assert complex(value, other) == complex(arr, other)
        except Exception as e:
            with pytest.raises(e.__class__):
                assert complex(value) == complex(arr)
                assert complex(value, other) == complex(arr, other)
        else:
            assert complex(value) == complex(arr)
            assert complex(value, other) == complex(arr, other)


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
            assert np_v == ap_v


@pytest.mark.parametrize('memmap', [True, False])
def test_math_redirects(tmpdir, memmap):
    f = os.path.join(tmpdir, 'redirects.npy')

    def check_arr(arr, a):
        arr_flags = arr.flags
        a_flags = a.flags
        for i in ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'WRITEABLE', 'ALIGNED',
                  'WRITEBACKIFCOPY', 'UPDATEIFCOPY', 'FNC', 'FORC',
                  'BEHAVED', 'CARRAY', 'FARRAY']:
            assert arr_flags[i] == a_flags[i]
        for i in ['OWNDATA']:
            if memmap:
                arr_flags[i] != a_flags[i]
            else:
                arr_flags[i] == a_flags[i]

        if memmap:
            assert isinstance(a.base, mmap.mmap)
        else:
            assert a.base is None

        assert arr.shape == a.shape
        assert arr.strides == a.strides
        assert arr.ndim == a.ndim
        assert arr.data == a.data
        assert arr.size == a.size
        assert arr.itemsize == a.itemsize
        assert arr.nbytes == a.nbytes
        assert arr.dtype == a.dtype

        assert isinstance(a.tolist(), list)
        assert arr.tolist() == a.tolist()
        assert isinstance(a.tostring(), bytes)
        assert arr.tostring() == a.tostring()
        assert isinstance(a.tobytes(), bytes)
        assert arr.tobytes() == a.tobytes()
        assert isinstance(a.dumps(), bytes)
        # FIXME: assert arr.dumps() == a.dumps()

        npt.assert_array_equal(arr.T, a.T)
        npt.assert_array_equal(arr.transpose(), a.transpose())
        npt.assert_array_equal(arr.flatten(), a.flatten())
        npt.assert_array_equal(arr.ravel(), a.ravel())
        npt.assert_array_equal(arr.squeeze(), a.squeeze())
        npt.assert_array_equal(arr.argsort(), a.argsort())
        npt.assert_array_equal(arr.argpartition(1), a.argpartition(1))
        npt.assert_array_equal(arr.nonzero(), a.nonzero())
        assert arr.max() == a.max()
        assert arr.argmax() == a.argmax()
        assert arr.min() == a.min()
        assert arr.argmin() == a.argmin()
        npt.assert_array_equal(arr.max(axis=0), a.max(axis=0))
        npt.assert_array_equal(arr.min(axis=0), a.min(axis=0))
        npt.assert_array_equal(arr.argmax(axis=0), a.argmax(axis=0))
        npt.assert_array_equal(arr.argmin(axis=0), a.argmin(axis=0))
        npt.assert_array_equal(arr.real, a.real)
        npt.assert_array_equal(arr.imag, a.imag)
        npt.assert_array_equal(arr.round(), a.round())
        assert arr.sum() == a.sum()
        npt.assert_array_equal(arr.sum(axis=0), a.sum(axis=0))
        npt.assert_array_equal(arr.cumsum(), a.cumsum())
        npt.assert_array_equal(arr.cumsum(axis=0), a.cumsum(axis=0))
        assert arr.mean() == a.mean()
        npt.assert_array_equal(arr.mean(axis=0), a.mean(axis=0))
        assert arr.var() == a.var()
        npt.assert_array_equal(arr.var(axis=0), a.var(axis=0))
        assert arr.std() == a.std()
        npt.assert_array_equal(arr.std(axis=0), a.std(axis=0))
        assert arr.prod() == a.prod()
        npt.assert_array_equal(arr.prod(axis=0), a.prod(axis=0))
        npt.assert_array_equal(arr.cumprod(), a.cumprod())
        npt.assert_array_equal(arr.cumprod(axis=0), a.cumprod(axis=0))

        for i, j in zip(arr.flat, a.flat):
            assert i == j
        for i in range(9):
            assert arr.item(i) == a.item(i)

        npt.assert_array_equal(arr.astype(bool), a.astype(bool))
        npt.assert_array_equal(arr.astype(int), a.astype(int))

        assert arr.all() == a.all()
        assert arr.any() == a.any()

        # FIXME: assert arr.ctypes == a.ctypes

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
