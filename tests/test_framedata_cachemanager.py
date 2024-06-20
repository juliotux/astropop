# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import os
import pytest
from astropop.framedata import cache_manager
from astropop.framedata.cache_manager import TempDir, BaseTempDir, TempFile

from astropop.testing import *


class Test_AtExit:
    def test_ensure_delete_on_exit(self):
        import atexit
        t = TempDir('testing')
        assert_in(t, BaseTempDir.managed.values())
        assert_in(BaseTempDir, cache_manager.managed_folders)
        assert_path_exists(t.full_path)
        assert_path_exists(BaseTempDir.full_path)

        atexit._run_exitfuncs()
        assert_path_not_exists(t.full_path)
        assert_path_not_exists(BaseTempDir.full_path)

    def test_keep_on_exit(self):
        import atexit
        cache_manager.DELETE_ON_EXIT = False
        t = TempDir('testing', delete_on_remove=False)
        assert_in(t, BaseTempDir.managed.values())
        assert_path_exists(t.full_path)
        assert_path_exists(BaseTempDir.full_path)

        atexit._run_exitfuncs()
        assert_path_exists(t.full_path)
        assert_path_exists(BaseTempDir.full_path)
        # restore the behavior
        cache_manager.DELETE_ON_EXIT = True


class Test_TempDir_Init:
    def test_init_relative_path(self):
        tmp = TempDir('testing')
        assert_equal(tmp.dirname, 'testing')
        assert_equal(tmp.full_path,
                     os.path.join(BaseTempDir.full_path, 'testing'))

    def test_init_absolute_path(self, tmpdir):
        tmp = TempDir(str(tmpdir))
        assert_equal(tmp.full_path, str(tmpdir))

    def test_init_creation_relative(self):
        tmp = TempDir('testing')
        assert_path_exists(tmp.full_path)

    def test_init_creation_absolute(self, tmpdir):
        tmp = TempDir(str(tmpdir))
        assert_path_exists(tmp.full_path)

    def test_init_error_parent_and_absolute(self, tmpdir):
        with pytest.raises(ValueError, match='Parent cannot be set for an '
                           'absolute dirname.'):
            TempDir(str(tmpdir), parent=BaseTempDir, delete_on_remove=False)

    def test_init_error_not_basename(self):
        with pytest.raises(ValueError, match='dirname must be a base name'):
            TempDir('testing/test', parent=BaseTempDir,
                    delete_on_remove=False)

    def test_init_error_invalid_parent(self):
        with pytest.raises(ValueError, match='a TempDir instance'):
            TempDir('testing', parent='testing', delete_on_remove=False)

class Test_TempDir_Methods:
    def test_create_folder(self):
        tmp = TempDir('testing')
        sub = tmp.create_folder('subfolder')
        assert_path_exists(sub.full_path)
        assert_equal(sub.full_path,
                     os.path.join(tmp.full_path, 'subfolder'))
        assert_is(sub.parent, tmp)

    def test_create_file(self):
        tmp = TempDir('testing')
        f = tmp.create_file('file.txt')
        assert_equal(f.full_path,
                     os.path.join(tmp.full_path, 'file.txt'))
        assert_is(f.parent, tmp)


class Test_TempDir_Delete:
    def test_delete(self):
        tmp = TempDir('testing')
        path = tmp.full_path
        assert_path_exists(path)
        tmp.delete()
        assert_path_not_exists(path)

    def test_keep_if_children_file(self):
        tmp = TempDir('testing')
        f = tmp.create_file('file.txt', delete_on_remove=False)
        path = tmp.full_path
        assert_path_exists(path)
        tmp.delete()
        assert_path_exists(path)

    def test_keep_if_children_folder(self):
        tmp = TempDir('testing')
        sub = tmp.create_folder('subfolder', delete_on_remove=False)
        path = tmp.full_path
        assert_path_exists(path)
        tmp.delete()
        assert_path_exists(path)

    def test_delete_children_multi_level(self):
        tmp = TempDir('testing')
        sub = tmp.create_folder('subfolder')
        f = sub.create_file('file.txt', delete_on_remove=False)
        tmp.delete()
        assert_path_exists(tmp.full_path)

class Test_TempDir_Attributes:
    def test_is_removable(self):
        tmp = TempDir('testing')
        assert_true(tmp.is_removable)

    def test_is_removable_multi_level(self):
        tmp = TempDir('testing')
        sub = tmp.create_folder('subfolder', delete_on_remove=True)
        f = sub.create_file('file.txt', delete_on_remove=False)
        assert_false(tmp.is_removable)
        assert_false(sub.is_removable)

    def test_str(self):
        assert_equal(str(BaseTempDir), BaseTempDir.full_path)

    def test_repr(self):
        assert_equal(repr(BaseTempDir),
                     f'<TempDir: "{BaseTempDir.full_path}" at'
                     f' {hex(id(BaseTempDir))}>')

    def test_context_manager(self):
        with TempDir('testing') as tmp:
            assert_path_exists(tmp.full_path)
            fpath = tmp.full_path
        assert_path_not_exists(fpath)


class Test_TempFile_Init:
    def test_init_relative_path(self):
        tmp = TempFile('testing')
        assert_equal(tmp.filename, 'testing')
        assert_equal(tmp.full_path,
                     os.path.join(BaseTempDir.full_path, 'testing'))

    def test_init_absolute_path(self, tmpdir):
        with pytest.raises(ValueError, match='filename must be a base name, '
                           'not a full path.'):
            TempFile(str(tmpdir))

    def test_delete(self):
        tmp = TempFile('testing')
        path = tmp.full_path
        f = tmp.open('w')
        f.close()
        tmp.delete()
        assert_path_not_exists(path)

    def test_init_error_not_basename(self):
        with pytest.raises(ValueError, match='filename must be a base name'):
            TempFile('testing/test', delete_on_remove=False)

    def test_init_error_invalid_parent(self):
        with pytest.raises(ValueError, match='a TempDir instance'):
            TempFile('testing', parent='testing', delete_on_remove=False)


class Test_TempFile_Methods:
    def test_open_close(self):
        tmp = TempFile('testing')
        f = tmp.open('w')
        assert_false(f.closed)
        assert_path_exists(tmp.full_path)
        f.write('testing')
        f.close()
        assert_true(f.closed)

        f = tmp.open('r')
        assert_false(f.closed)
        assert_equal(f.read(), 'testing')
        f.close()
        assert_true(f.closed)

    def test_delete(self):
        tmp = TempFile('testing')
        f = tmp.open('w')
        f.write('testing')
        f.close()
        path = tmp.full_path
        tmp.delete()
        assert_path_not_exists(path)

    def test_context_manager(self):
        with TempFile('testing') as tmp:
            fpath = tmp.full_path
            f = tmp.open('w')
            f.write('testing')
            f.close()
            assert_path_exists(tmp.full_path)
            f = tmp.open('r')
            assert_equal(f.read(), 'testing')

        assert_path_not_exists(fpath)
        assert_true(f.closed)


class Test_TempFile_Attributes:
    def test_is_removable(self):
        tmp = TempFile('testing')
        assert_true(tmp.is_removable)
        tmp = TempFile('testing', delete_on_remove=False)
        assert_false(tmp.is_removable)

    def test_str(self):
        tmp = TempFile('testing')
        assert_equal(str(tmp), tmp.full_path)

    def test_repr(self):
        tmp = TempFile('testing')
        assert_equal(repr(tmp),
                     f'<TempFile: "{tmp.full_path}" at {hex(id(tmp))}>')
