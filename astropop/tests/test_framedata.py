# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import tempfile
import os
import numpy as np
import numpy.testing as npt
from astropop.framedata import FrameData, create_array_memmap, \
                               delete_array_memmap, ensure_bool_mask, \
                               setup_filename


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
    b = delete_array_memmap(b)
    c = delete_array_memmap(c)
    assert not isinstance(b, np.memmap)
    assert not isinstance(b, np.memmap)
    assert isinstance(b, np.ndarray)
    assert isinstance(c, np.ndarray)
    npt.assert_array_equal(a, b)
    npt.assert_allclose(a, c)
    assert not os.path.exists(f)
    assert not os.path.exists(g)

    # None should not raise errors
    create_array_memmap('dummy', None)
    delete_array_memmap('dummy', None)


def test_ensure_bool_mask(tmpdir):
    # Bool array
    b_array = np.zeros(2, dtype=bool)
    mask = ensure_bool_mask(b_array)
    assert np.dtype(mask.dtype) is np.dtype(bool)
    npt.assert_array_equal(mask, b_array)

    # Integer array
    i_array = np.zeros(2, dtype='i4')
    mask = ensure_bool_mask(i_array)
    assert np.dtype(mask.dtype) is np.dtype(bool)
    npt.assert_array_almost_equal(mask, i_array)

    # Float array
    f_array = np.zeros(2, dtype='f8')
    mask = ensure_bool_mask(f_array)
    assert np.dtype(mask.dtype) is np.dtype(bool)
    npt.assert_array_almost_equal(mask, f_array)

    # Memmap
    filename = tmpdir.join('memmap.npy')
    m_array = np.memmap(filename.open(), shape=2, dtype='i4', mode='w+')
    mask = ensure_bool_mask(m_array)
    assert np.dtype(mask.dtype) is np.dtype(bool)
    npt.assert_array_almost_equal(mask, m_array)
    del m_array
    os.remove(filename)


def test_setup_filename(tmpdir):
    temp = os.path.abspath(tmpdir)
    fname = 'test_filename.npy'
    test_obj = FrameData(np.zeros(2), unit='adu',
                         cache_filename = 'test_filename.npy',
                         cache_folder = temp)
    
    assert setup_filename(test_obj) == os.path.join(temp, fname)
    assert tmpdir.exists()
    # Manual set filename
    ntemp = tempfile.mkstemp(suffix='.npy')[1]
    # with obj and manual filename, keep object
    assert setup_filename(test_obj, filename=ntemp) == os.path.join(temp, fname)
    test_obj.cache_filename = None
    assert setup_filename(test_obj, filename=ntemp) == os.path.join(temp, os.path.basename(ntemp))
    # same for cache folder
    test_obj.cache_filename = fname
    assert setup_filename(test_obj, filename=ntemp) == os.path.join(temp, fname)
    test_obj.cache_folder = None
    cache = '/tmp/astropop_testing'
    assert setup_filename(test_obj, cache_folder=cache) == os.path.join(cache, fname)
    assert os.path.isdir(cache)
    os.removedirs(cache)

    # now, with full random
    test_obj.cache_filename = None
    test_obj.cache_folder = None
    sfile = setup_filename(test_obj)
    dirname = os.path.dirname(sfile)
    filename = os.path.basename(sfile)
    assert dirname == test_obj.cache_folder
    assert filename == test_obj.cache_filename
    assert os.path.exists(dirname)
