# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Manage owned temporary files without removing unrelated user files."""

import os
import tempfile
import weakref


__all__ = ['CacheManager']


def _cleanup(path, files, remove_directory):
    """Clean up only files reserved by this manager; safe to call repeatedly."""
    for basename in list(files):
        filename = os.path.join(path, basename)
        try:
            stat = os.stat(filename, follow_symlinks=False)
            if (stat.st_dev, stat.st_ino) == files[basename]:
                os.remove(filename)
        except FileNotFoundError:
            pass
        del files[basename]
    if remove_directory:
        try:
            os.rmdir(path)
        except (FileNotFoundError, OSError):
            # Never recursively delete a folder: it may contain unrelated files.
            pass


class CacheManager:
    """Own temporary files in a cache directory.

    Existing directories are preserved. Only files created by ``add_file``
    are removed, at cleanup or garbage collection (and at process exit).
    ``delete_on_exit=False`` retains both the directory and its files.
    """

    def __init__(self, cache_folder=None, delete_on_exit=True):
        if cache_folder is None:
            cache_folder = tempfile.mkdtemp(prefix='astropop-cache-')
            owns_directory = True
        else:
            cache_folder = os.path.abspath(os.fspath(cache_folder))
            try:
                os.makedirs(cache_folder)
                owns_directory = True
            except FileExistsError:
                if not os.path.isdir(cache_folder):
                    raise
                owns_directory = False
        self._cache_folder = os.path.abspath(cache_folder)
        self._files = {}
        self._delete_on_exit = delete_on_exit
        self._finalizer = weakref.finalize(
            self, _cleanup, self.path, self._files, owns_directory)
        if not delete_on_exit:
            self._finalizer.detach()

    @property
    def path(self):
        """Absolute cache directory path."""
        return self._cache_folder

    @property
    def managed_files(self):
        """Basenames of files reserved by this manager."""
        return list(self._files)

    @property
    def listdir(self):
        """All entries in the cache directory."""
        return os.listdir(self.path)

    def add_file(self, basename):
        """Reserve a file, refusing to overwrite an existing unowned file."""
        basename = os.fspath(basename)
        if basename in ('', '.', '..') or os.path.basename(basename) != basename:
            raise ValueError('basename must be a file name, not a path')
        path = os.path.join(self.path, basename)
        if basename in self._files and os.path.lexists(path):
            stat = os.stat(path, follow_symlinks=False)
            if (stat.st_dev, stat.st_ino) != self._files[basename]:
                raise FileExistsError(f'Cache file was replaced: {path}')
            return path
        with open(path, 'xb') as stream:
            stat = os.fstat(stream.fileno())
            self._files[basename] = (stat.st_dev, stat.st_ino)
        return path

    def remove_file(self, basename):
        """Remove a managed file, including its ownership record."""
        if basename not in self._files:
            raise ValueError(f'File not in cache folder: {basename}')
        _cleanup(self.path, {basename: self._files.pop(basename)}, False)

    def _cleanup(self):
        """Release owned files once, unless retention was requested."""
        self._finalizer()

    def __str__(self):
        return self.path

    def __repr__(self):
        return f'<CacheManager: {self.path}>'
