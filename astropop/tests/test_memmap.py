# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import pytest
import tempfile
from astropop.memmap import MemMapArray, array_bi, array_attr, create_array_memmap, delete_array_memmap
import numpy as np
import numpy.testing as npt


# TODO: Tests needed


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

