# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import pytest
import tempfile
from astropop.memmap import MemMapArray, array_bi, array_attr, create_array_memmap, delete_array_memmap
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

# TODO: Numpy functions
# TODO: units
# TODO: flush
# TODO: reset_data
# TODO: repr

################################################################################
# For math tests we supose that numpy's math is correct
################################################################################

@pytest.mark.parametrize('memmap, value', [(True, 2), (False, 2),
                                           (True, 3.5), (False, 3.5),
                                           (True, 0), (False, 0),
                                           (True, 20), (False, 20),
                                           (True, -10), (False, -10)])
def test_memmap_lt(tmpdir, memmap, value):
    if memmap:
        f = os.path.join(tmpdir, 'lt.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a < value, arr < value)


@pytest.mark.parametrize('memmap, value', [(True, 2), (False, 2),
                                           (True, 3.5), (False, 3.5),
                                           (True, 0), (False, 0),
                                           (True, 20), (False, 20),
                                           (True, -10), (False, -10)])
def test_memmap_le(tmpdir, memmap, value):
    if memmap:
        f = os.path.join(tmpdir, 'le.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a <= value, arr <= value)


@pytest.mark.parametrize('memmap, value', [(True, 2), (False, 2),
                                           (True, 3.5), (False, 3.5),
                                           (True, 0), (False, 0),
                                           (True, 20), (False, 20),
                                           (True, -10), (False, -10)])
def test_memmap_gt(tmpdir, memmap, value):
    if memmap:
        f = os.path.join(tmpdir, 'gt.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a > value, arr > value)


@pytest.mark.parametrize('memmap, value', [(True, 2), (False, 2),
                                           (True, 3.5), (False, 3.5),
                                           (True, 0), (False, 0),
                                           (True, 20), (False, 20),
                                           (True, -10), (False, -10)])
def test_memmap_ge(tmpdir, memmap, value):
    if memmap:
        f = os.path.join(tmpdir, 'ge.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a >= value, arr >= value)


@pytest.mark.parametrize('memmap, value', [(True, 2), (False, 2),
                                           (True, 3.5), (False, 3.5),
                                           (True, 0), (False, 0),
                                           (True, 20), (False, 20),
                                           (True, -10), (False, -10)])
def test_memmap_eq(tmpdir, memmap, value):
    if memmap:
        f = os.path.join(tmpdir, 'eq.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a == value, arr == value)


@pytest.mark.parametrize('memmap, value', [(True, 2), (False, 2),
                                           (True, 3.5), (False, 3.5),
                                           (True, 0), (False, 0),
                                           (True, 20), (False, 20),
                                           (True, -10), (False, -10)])
def test_memmap_ne(tmpdir, memmap, value):
    if memmap:
        f = os.path.join(tmpdir, 'ne.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1)
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a != value, arr != value)


@pytest.mark.parametrize('memmap,value,other', [(True, 10, 3), (False, 10, 3),
                                                (True, 10.5, 3), (False, 10.5, 3),
                                                (True, 20, 3.5), (False, 20, 3.5),
                                                (True, 20, 3), (False, 20, 3),
                                                (True, 20, -3), (False, 20, -3)])
def test_math_add_number(tmpdir, memmap, value, other):
    if memmap:
        f = os.path.join(tmpdir, 'add.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a+other, arr+other)   


@pytest.mark.parametrize('memmap,value,other', [(True, 10, 3), (False, 10, 3),
                                                (True, 10.5, 3), (False, 10.5, 3),
                                                (True, 20, 3.5), (False, 20, 3.5),
                                                (True, 20, 3), (False, 20, 3),
                                                (True, 20, -3), (False, 20, -3)])
def test_math_sub_number(tmpdir, memmap, value, other):
    if memmap:
        f = os.path.join(tmpdir, 'sun.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a-other, arr-other)


@pytest.mark.parametrize('memmap,value,other', [(True, 10, 3), (False, 10, 3),
                                                (True, 10.5, 3), (False, 10.5, 3),
                                                (True, 20, 3.5), (False, 20, 3.5),
                                                (True, 20, 3), (False, 20, 3),
                                                (True, 20, -3), (False, 20, -3)])
def test_math_pow_number(tmpdir, memmap, value, other):
    if memmap:
        f = os.path.join(tmpdir, 'pow.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a**other, arr**other)   


@pytest.mark.parametrize('memmap,value,other', [(True, 10, 3), (False, 10, 3),
                                                (True, 10.5, 3), (False, 10.5, 3),
                                                (True, 20, 3.5), (False, 20, 3.5),
                                                (True, 20, 3), (False, 20, 3),
                                                (True, 20, -3), (False, 20, -3)])
def test_math_truediv_number(tmpdir, memmap, value, other):
    if memmap:
        f = os.path.join(tmpdir, 'div.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a/other, arr/other)


@pytest.mark.parametrize('memmap,value,other', [(True, 10, 3), (False, 10, 3),
                                                (True, 10.5, 3), (False, 10.5, 3),
                                                (True, 20, 3.5), (False, 20, 3.5),
                                                (True, 20, 3), (False, 20, 3),
                                                (True, 20, -3), (False, 20, -3)])
def test_math_floordiv_number(tmpdir, memmap, value, other):
    if memmap:
        f = os.path.join(tmpdir, 'floordiv.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a//other, arr//other)   


@pytest.mark.parametrize('memmap,value,other', [(True, 10, 3), (False, 10, 3),
                                                (True, 10.5, 3), (False, 10.5, 3),
                                                (True, 20, 3.5), (False, 20, 3.5),
                                                (True, 20, 3), (False, 20, 3),
                                                (True, 20, -3), (False, 20, -3)])
def test_math_mul_number(tmpdir, memmap, value, other):
    if memmap:
        f = os.path.join(tmpdir, 'mul.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a*other, arr*other)


@pytest.mark.parametrize('memmap,value,other', [(True, 10, 3), (False, 10, 3),
                                                (True, 10.5, 3), (False, 10.5, 3),
                                                (True, 20, 3.5), (False, 20, 3.5),
                                                (True, 20, 3), (False, 20, 3),
                                                (True, 20, -3), (False, 20, -3)])
def test_math_mod_number(tmpdir, memmap, value, other):
    if memmap:
        f = os.path.join(tmpdir, 'mod.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(a%other, arr%other)


@pytest.mark.parametrize('memmap,value,other,raises', [(True, 3, 1, False), (False, 3, 1, False),
                                                       (True, 2, 3, False), (False, 2, 3, False)])
def test_math_lshift_number(tmpdir, memmap, value, other, raises):
    # TODO: Implement raises
    if memmap:
        f = os.path.join(tmpdir, 'lshift.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(arr<<other, a<<other)
    # FIXME: why do this simply don't work like in numpy array???
    # npt.assert_array_equal(other<<arr, other<<a)


@pytest.mark.parametrize('memmap,value,other,raises', [(True, 3, 1, False), (False, 3, 1, False),
                                                       (True, 2, 3, False), (False, 2, 3, False)])
def test_math_rshift_number(tmpdir, memmap, value, other, raises):
    # TODO: Implement raises
    if memmap:
        f = os.path.join(tmpdir, 'rshift.npy')
    else:
        f = None
    arr = np.arange(0, 10, 1) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(arr>>other, a>>other)
    # FIXME: why do this simply don't work like in numpy array???
    # npt.assert_array_equal(other>>arr, other>>a)


@pytest.mark.parametrize('memmap,value,other', [(True, 1, 1), (False, 1, 1),
                                                (True, 2, 3), (False, 2, 3),
                                                (True, 1, [[1, 0, 1, 1], [0, 1, 0, 0]]),
                                                (False, 1,  [[1, 0, 1, 1], [0, 1, 0, 0]])])
def test_math_and_number(tmpdir, memmap, value, other):
    if memmap:
        f = os.path.join(tmpdir, 'and.npy')
    else:
        f = None
    arr = np.array([[0, 1, 0, 1], [1, 0, 1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(arr & other, a & other)
    # FIXME: why do this simply don't work like in numpy array???
    # npt.assert_array_equal(other & arr, other & a)


@pytest.mark.parametrize('memmap,value,other', [(True, 1, 1), (False, 1, 1),
                                                (True, 2, 3), (False, 2, 3),
                                                (True, 1, [[1, 0, 1, 1], [0, 1, 0, 0]]),
                                                (False, 1,  [[1, 0, 1, 1], [0, 1, 0, 0]])])
def test_math_or_number(tmpdir, memmap, value, other):
    if memmap:
        f = os.path.join(tmpdir, 'xor.npy')
    else:
        f = None
    arr = np.array([[0, 1, 0, 1], [1, 0, 1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(arr | other, a | other)
    # FIXME: why do this simply don't work like in numpy array???
    # npt.assert_array_equal(other | arr, other | a)


@pytest.mark.parametrize('memmap,value,other', [(True, 1, 1), (False, 1, 1),
                                                (True, 2, 3), (False, 2, 3),
                                                (True, 1, [[1, 0, 1, 1], [0, 1, 0, 0]]),
                                                (False, 1,  [[1, 0, 1, 1], [0, 1, 0, 0]])])
def test_math_xor_number(tmpdir, memmap, value, other):
    if memmap:
        f = os.path.join(tmpdir, 'xor.npy')
    else:
        f = None
    arr = np.array([[0, 1, 0, 1], [1, 0, 1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(arr ^ other, a ^ other)
    # FIXME: why do this simply don't work like in numpy array???
    # npt.assert_array_equal(other | arr, other | a)


@pytest.mark.parametrize('memmap,value', [(True, 1), (False, 1),
                                          (True, 2.5), (False, 2.5),
                                          (True, -1), (False, -1),
                                          (True, -2.5), (False, -2.5)])
def test_math_neg_number(tmpdir, memmap, value):
    if memmap:
        f = os.path.join(tmpdir, 'neg.npy')
    else:
        f = None
    arr = np.array([[0, 1, 0, -1], [1, 0, -1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(-arr, -a)


@pytest.mark.parametrize('memmap,value', [(True, 1), (False, 1),
                                          (True, 2.5), (False, 2.5),
                                          (True, -1), (False, -1),
                                          (True, -2.5), (False, -2.5)])
def test_math_pos_number(tmpdir, memmap, value):
    if memmap:
        f = os.path.join(tmpdir, 'pos.npy')
    else:
        f = None
    arr = np.array([[0, 1, 0, -1], [1, 0, -1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(+arr, +a)



@pytest.mark.parametrize('memmap,value', [(True, 1), (False, 1),
                                          (True, 2.5), (False, 2.5),
                                          (True, -1), (False, -1),
                                          (True, -2.5), (False, -2.5)])
def test_math_abs_number(tmpdir, memmap, value):
    if memmap:
        f = os.path.join(tmpdir, 'abs.npy')
    else:
        f = None
    arr = np.array([[0, 1, 0, -1], [1, 0, -1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(arr.__abs__(), a.__abs__())


@pytest.mark.parametrize('memmap,value', [(True, 1), (False, 1),
                                          (True, -1), (False, -1)])
def test_math_invert_number(tmpdir, memmap, value):
    if memmap:
        f = os.path.join(tmpdir, 'invert.npy')
    else:
        f = None
    arr = np.array([[0, 1, 0, -1], [1, 0, -1, 0]]) * value
    a = MemMapArray(arr, filename=f, memmap=memmap)
    npt.assert_array_equal(~arr, ~a)


# TODO: matmul
# TODO: bool
# TODO: i<add, sub, ...>
# TODO: len, contains, int, float, complex
# TODO: array_attr