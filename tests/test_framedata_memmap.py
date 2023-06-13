# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import os
import pytest
from astropop.framedata._memmap import create_array_memmap, \
                                       delete_array_memmap, \
                                       reset_memmap_array
import numpy as np

from astropop.testing import *


class TestMemmapCreation:
    a = np.ones((30, 30), dtype='i4')
    b = np.ones((30, 30), dtype='f4')*2
    c = np.ones((30, 30), dtype='f8')*3.14

    def test_create_memmap_none_filename(self):
        with pytest.raises(ValueError, match='None filename'):
            create_array_memmap(None, self.a)

    def test_create_memmap_simple(self, tmpdir):
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, self.a)
        assert_is_instance(b, np.memmap)
        assert_equal(self.a, b)
        assert_equal(b.dtype, self.a.dtype)
        assert_true(os.path.exists(f))

    def test_create_memmap_dtype(self, tmpdir):
        # numpy dtype
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, self.c, dtype='f4')
        assert_is_instance(b, np.memmap)
        assert_almost_equal(self.c, b)
        assert_equal(b.dtype, np.dtype('f4'))
        assert_true(os.path.exists(f))

        # python dtype
        f = os.path.join(tmpdir, 'testarray2.npy')
        b = create_array_memmap(f, self.a, dtype=bool)
        assert_is_instance(b, np.memmap)
        assert_equal(self.a, b)
        assert_equal(b.dtype, bool)

    def test_create_memmap_none(self, tmpdir):
        # None should not be memmaped
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, None)
        assert_is_none(b)
        assert_is_not_instance(b, np.memmap)
        assert_false(os.path.exists(f))

    def test_create_memmap_scalar(self, tmpdir):
        # Scalar should not be memmaped
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, 1)
        assert_is_not_instance(b, np.ndarray)
        assert_equal(b, 1)
        assert_false(os.path.exists(f))


class TestMemmapDeletion:
    a = np.ones((30, 30), dtype='i4')

    def test_delete_memmap_read_no_remove(self, tmpdir):
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, self.a)
        # Deletion
        d = delete_array_memmap(b, read=True, remove=False)
        assert_is_not_instance(d, np.memmap)
        assert_is_instance(d, np.ndarray)
        assert_equal(self.a, d)
        assert_true(os.path.exists(f))

    def test_delete_memmap_read_remove(self, tmpdir):
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, self.a)
        # Deletion
        d = delete_array_memmap(b, read=True, remove=True)
        assert_is_not_instance(d, np.memmap)
        assert_is_instance(d, np.ndarray)
        assert_equal(self.a, d)
        assert_false(os.path.exists(f))

    def test_delete_memmap_no_read_no_remove(self, tmpdir):
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, self.a)
        # Deletion
        d = delete_array_memmap(b, read=False, remove=False)
        assert_is_none(d)
        assert_true(os.path.exists(f))

    def test_delete_memmap_no_read_remove(self, tmpdir):
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, self.a)
        # Deletion
        d = delete_array_memmap(b, read=False, remove=True)
        assert_is_none(d)
        assert_false(os.path.exists(f))

    def test_delete_memmap_none(self):
        d = delete_array_memmap(None)
        assert_is_none(d)

    def test_delete_memmap_scalar(self):
        d = delete_array_memmap(1)
        assert_equal(d, 1)


class TestMemmapReset:
    a = np.ones((30, 30), dtype='i4')
    b = np.ones((30, 30), dtype='f4')*2

    def test_reset_memmap(self, tmpdir):
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, self.a)
        b = reset_memmap_array(b, self.b)
        assert_equal(b, self.b)
        assert_equal(b.dtype, self.b.dtype)
        assert_true(os.path.exists(f))

    def test_reset_memmap_none(self, tmpdir):
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, self.a)
        b = reset_memmap_array(b, None)
        assert_is_none(b)
        assert_false(os.path.exists(f))

    def test_reset_memmap_scalar(self, tmpdir):
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, self.a)
        b = reset_memmap_array(b, 1)
        assert_is_not_instance(b, np.ndarray)
        assert_equal(b, 1)
        assert_false(os.path.exists(f))

    def test_reset_memmap_dtype(self, tmpdir):
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, self.a)
        b = reset_memmap_array(b, self.b, dtype='f8')
        assert_equal(b, self.b)
        assert_equal(b.dtype, np.dtype('f8'))
        assert_true(os.path.exists(f))

    @pytest.mark.parametrize('order', ['<', '>', '='])
    def test_reset_memmap_dtype_nonative(self, tmpdir, order):
        f = os.path.join(tmpdir, 'testarray.npy')
        b = create_array_memmap(f, self.a.astype(order+'f4'))
        b = reset_memmap_array(b, self.b)
        assert_equal(b, self.b)
        assert_true(b.dtype.isnative)
        assert_true(os.path.exists(f))
        delete_array_memmap(b, read=False, remove=True)

        b = reset_memmap_array(b, self.b, dtype=order+'f8')
        assert_equal(b, self.b)
        assert_true(b.dtype.isnative)
        assert_true(os.path.exists(f))
