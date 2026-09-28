"""Cache ownership, cleanup, and explicit FrameData memory mapping."""

import gc
from pathlib import Path
import weakref

import numpy as np
import pytest

from astropop.config import AstropopConfig as conf
from astropop.framedata import FrameData
from astropop.framedata._cache_manager import CacheManager


def test_cache_cleanup_preserves_user_files(tmp_path):
    unrelated = tmp_path/'important.fits'
    unrelated.write_bytes(b'user data')
    manager = CacheManager(tmp_path)
    for i in range(4):
        Path(manager.add_file(f'frame{i}')).write_bytes(b'cache data')
    manager._cleanup()
    manager._cleanup()
    assert unrelated.read_bytes() == b'user data'
    assert list(tmp_path.iterdir()) == [unrelated]


def test_cache_ownership_and_missing_files(tmp_path):
    manager = CacheManager(tmp_path/'new')
    path = Path(manager.add_file('frame'))
    path.unlink()
    manager._cleanup()
    assert not (tmp_path/'new').exists()


def test_cache_collected_without_atexit_reference(tmp_path):
    manager = CacheManager(tmp_path/'new')
    path = Path(manager.add_file('frame'))
    ref = weakref.ref(manager)
    del manager
    gc.collect()
    assert ref() is None
    assert not path.exists()


def test_cache_retain(tmp_path):
    manager = CacheManager(tmp_path, delete_on_exit=False)
    path = Path(manager.add_file('frame'))
    del manager
    gc.collect()
    assert path.exists()


@pytest.mark.parametrize('name', ['', '.', '..', '../data', '/tmp/data'])
def test_cache_reject_paths(tmp_path, name):
    manager = CacheManager(tmp_path)
    with pytest.raises(ValueError):
        manager.add_file(name)


def test_cache_refuses_collision_and_symlink(tmp_path):
    protected = tmp_path/'data'
    protected.write_bytes(b'untouched')
    (tmp_path/'link').symlink_to(protected)
    manager = CacheManager(tmp_path)
    for name in ['data', 'link']:
        with pytest.raises(FileExistsError):
            manager.add_file(name)
    manager._cleanup()
    assert protected.read_bytes() == b'untouched'


def test_cache_replaced_file_is_not_removed(tmp_path):
    manager = CacheManager(tmp_path)
    original = Path(manager.add_file('data'))
    replacement = tmp_path/'replacement'
    replacement.write_bytes(b'new owner')
    replacement.replace(original)
    with pytest.raises(FileExistsError):
        manager.add_file('data')
    manager._cleanup()
    assert original.read_bytes() == b'new owner'


def test_frame_requires_explicit_cache():
    frame = FrameData(np.ones((2, 3)))
    assert frame.cache is None
    with pytest.raises(ValueError, match='cache_folder'):
        frame.enable_memmap()
    with pytest.raises(ValueError, match='cache_folder'):
        FrameData(np.ones((2, 3)), use_memmap_backend=True)


def test_frame_copies_have_independent_cache_files(tmp_path):
    original = FrameData(np.ones((3, 4)), uncertainty=2,
                         cache_folder=tmp_path, use_memmap_backend=True)
    first, second = original.copy(), original.copy()
    assert len({f.data.filename for f in [original, first, second]}) == 3
    first.data[:] = 4
    np.testing.assert_array_equal(second.data, 1)
    files = [Path(first.data.filename), Path(second.data.filename)]
    del original
    gc.collect()
    assert all(p.exists() for p in files)
    del first, second
    gc.collect()
    assert not list(tmp_path.iterdir())


def test_disabling_storage_releases_memmap_files(monkeypatch, tmp_path):
    frame = FrameData(np.ones((3, 4)), uncertainty=2,
                      cache_folder=tmp_path, use_memmap_backend=True)
    unct, flags = Path(frame.uncertainty.filename), Path(frame.flags.filename)
    monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_UNCERTAINTY', True)
    monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_FLAGS', True)
    frame.uncertainty = 3
    frame.flags = np.zeros(frame.shape)
    assert frame.flags is frame.uncertainty is None
    assert not unct.exists() and not flags.exists()
    assert Path(frame.data.filename).exists()


def test_memmap_updates_and_reenable(tmp_path):
    frame = FrameData(np.ones((3, 4)), uncertainty=2,
                      cache_folder=tmp_path, use_memmap_backend=True)
    frame.data = np.full(frame.shape, 5)
    frame.flags = np.ones(frame.shape)
    frame.uncertainty = 3
    frame.disable_memmap()
    assert not list(tmp_path.iterdir())
    frame.enable_memmap()
    np.testing.assert_array_equal(frame.data, 5)
    np.testing.assert_array_equal(frame.flags, 1)
    np.testing.assert_array_equal(frame.uncertainty, 3)
    del frame
    gc.collect()
    assert not list(tmp_path.iterdir())


def test_default_cache_directory_lifecycle():
    manager = CacheManager()
    directory = Path(manager.path)
    assert directory.is_dir()
    assert str(manager) == str(directory)
    assert str(directory) in repr(manager)
    manager._cleanup()
    assert not directory.exists()


def test_cache_rejects_file_as_directory(tmp_path):
    path = tmp_path/'file'
    path.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        CacheManager(path)
    assert path.read_bytes() == b'preserve'


def test_cache_owned_directory_with_unrelated_file(tmp_path):
    manager = CacheManager(tmp_path/'cache')
    unrelated = Path(manager.path)/'user.fits'
    unrelated.write_bytes(b'preserve')
    manager.add_file('temporary')
    manager._cleanup()
    assert unrelated.read_bytes() == b'preserve'
    assert list(unrelated.parent.iterdir()) == [unrelated]


def test_cache_explicit_removal_and_reservation(tmp_path):
    manager = CacheManager(tmp_path)
    filename = manager.add_file('data')
    Path(filename).write_bytes(b'unchanged')
    assert manager.add_file('data') == filename
    assert Path(filename).read_bytes() == b'unchanged'
    managed = manager.managed_files
    managed.clear()
    assert manager.managed_files == ['data']
    assert manager.listdir == ['data']
    manager.remove_file('data')
    assert manager.managed_files == manager.listdir == []
    with pytest.raises(ValueError, match='not in cache'):
        manager.remove_file('unowned')


def test_cleanup_does_not_run_other_exit_handlers(tmp_path):
    import atexit

    calls = []
    callback = lambda: calls.append('unrelated handler')
    atexit.register(callback)
    try:
        manager = CacheManager(tmp_path)
        manager.add_file('data')
        manager._cleanup()
        assert not calls
    finally:
        atexit.unregister(callback)


@pytest.mark.parametrize('retain', [False, True])
def test_cache_cleanup_on_process_exit(tmp_path, retain):
    import subprocess
    import sys

    # Never call atexit._run_exitfuncs() in the pytest process: it also stops
    # coverage measurement and runs unrelated libraries' shutdown handlers.
    script = '''
import sys
from astropop.framedata._cache_manager import CacheManager
manager = CacheManager(sys.argv[1], delete_on_exit=sys.argv[2] == 'False')
manager.add_file('data')
'''
    directory = tmp_path/'cache'
    subprocess.run([sys.executable, '-c', script, str(directory), str(retain)],
                   check=True, capture_output=True, text=True)
    assert directory.exists() == retain
    assert (directory/'data').exists() == retain
