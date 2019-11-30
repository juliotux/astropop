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
    assert a.unit is None
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


# TODO: Math operations tests
# TODO: Numpy functions
# TODO: units
# TODO: flush
# TODO: reset_data
# TODO: repr
