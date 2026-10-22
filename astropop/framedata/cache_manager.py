# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Managing the cache folder for FrameData memmaping."""

from __future__ import annotations

import typing as _t
import types as _types

import os
import atexit
import shutil

from astropy.config import get_cache_dir


from ..logger import logger


__all__ = ['TempFile', 'TempDir', 'BaseTempDir', 'cleanup',
           'DELETE_ON_EXIT', 'CACHE_DIR']


# keep all the managed folders here to delete on exit.
# elements are TempDir instances
managed_folders = []
DELETE_ON_EXIT = True
CACHE_DIR = os.path.join(get_cache_dir(), 'astropop')


def cleanup() -> None:
    """Remove the cache folder."""
    logger.debug('Cleaning up cache folder: %s', CACHE_DIR)
    if DELETE_ON_EXIT:
        for i in managed_folders:
            i.delete()


atexit.register(cleanup)


class TempDir:
    """A temporary directory that will be deleted on exit.

    This is a context manager that will create a temporary directory in the
    cache folder. The directory will be deleted on exit.

    Parameters
    ----------
    dirname
        The directory name to be created.
    delete_on_remove
        If True, the directory will be deleted when removed from the manager.
    parent
        Parent manager for a relative basename. With None, relative names use
        BaseTempDir and absolute names create top-level managed directories.

    Examples
    --------
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from astropop.framedata.cache_manager import TempDir
    >>> with TemporaryDirectory() as root:
    ...     # Give the manager its own child directory because it owns cleanup.
    ...     with TempDir(str(Path(root) / 'owned-cache')) as cache:
    ...         managed = cache.create_file('message.txt')
    ...         with managed.open('w') as stream:
    ...             _ = stream.write('example')
    ...         print(Path(managed.full_path).read_text())
    ...     print(Path(cache.full_path).exists())
    example
    False
    """

    __slots__ = ['dirname', 'parent', 'managed', 'delete_on_remove']

    def __init__(
        self,
        dirname: str,
        parent: TempDir | None=None,
        delete_on_remove: bool=True,
    ) -> None:
        """Create the directory and register it with its parent or the global manager."""
        is_absolute = os.path.isabs(dirname)

        # only accept relative parents
        if parent is not None and is_absolute:
            raise ValueError('Parent cannot be set for an absolute dirname.')
        if parent is None and not is_absolute:
            # for relative, non-parent directories, use the base cache folder
            parent = BaseTempDir
        if parent is not None and os.path.basename(dirname) != dirname:
            raise ValueError(f'{dirname} dirname must be a base name'
                             ' if not absolute.')

        # initialize slots
        self.managed = {} # key: basename, value: TempFile or TempDir
        self.delete_on_remove = delete_on_remove
        self.dirname = dirname

        # for absolute directories, add to managed_folders
        # for relative directories with a parent, use the parent
        if is_absolute:
            self.parent = None
            managed_folders.append(self)
        elif isinstance(parent, TempDir):
            self.parent = parent
            if self not in self.parent.managed.values():
                self.parent.managed[self.dirname] = self
        else:
            raise ValueError('parent must be None or a TempDir instance.')

        # automatically create the directory
        os.makedirs(self.full_path, exist_ok=True)

    @property
    def full_path(self) -> str:
        """Absolute directory path, resolving a relative name through its parent."""
        if self.parent is None:
            return self.dirname
        return os.path.join(self.parent.full_path, self.dirname)

    def create_file(self, basename: str, delete_on_remove: bool=True) -> TempFile:
        """Add a TempFile to this folder.

        Parameters
        ----------
        basename
            Base name of the file to be created.
        delete_on_remove
            Allow deletion on removal or shutdown when True. False retains it.

        Returns
        -------
        TempFile
            The created TempFile instance.
        """
        f = TempFile(basename, parent=self,
                     delete_on_remove=delete_on_remove)
        self.managed[basename] = f
        return f

    def create_folder(self, basename: str, delete_on_remove: bool=True) -> TempDir:
        """Add a TempFolder to this folder.

        Parameters
        ----------
        basename
            Base name of the file or folder to be created.
        delete_on_remove
            Allow deletion on removal or shutdown when True. False retains it.

        Returns
        -------
        TempDir
            The created TempDir instance.
        """
        f = TempDir(basename, parent=self,
                    delete_on_remove=delete_on_remove)
        self.managed[basename] = f
        return f

    def delete(self) -> None:
        """Delete this folder and all its contents."""
        if self.is_removable:
            # remove all the children first
            childrend = list(self.managed.values())
            for i in childrend:
                i.delete()
            del childrend

            # remove from parent or managed folders
            if self in managed_folders:
                managed_folders.remove(self)
            if self.parent is not None:
                try:
                    del self.parent.managed[self.dirname]
                except KeyError:
                    pass
            try:
                shutil.rmtree(self.full_path)
            except FileNotFoundError:
                pass

    @property
    def is_removable(self) -> bool:
        """This folder can be removed and all its contents."""
        keep = not self.delete_on_remove
        for i in self.managed.values():
            keep = keep or not i.is_removable
        return not keep

    def __enter__(self) -> _t.Self:
        """Return this directory manager for use inside a context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: _types.TracebackType | None,
    ) -> None:
        """Apply directory deletion policy on exit without suppressing exceptions."""
        self.delete()

    def __str__(self) -> str:
        """Return the configured directory name."""
        return self.dirname

    def __repr__(self) -> str:
        """Identify the directory manager by its name and object address."""
        return f'<TempDir: "{self.dirname}" at {hex(id(self))}>'


# Base cache folder
BaseTempDir = object.__new__(TempDir)
BaseTempDir.dirname = CACHE_DIR
BaseTempDir.parent = None
BaseTempDir.managed = {}
BaseTempDir.delete_on_remove = True
managed_folders.append(BaseTempDir)
os.makedirs(CACHE_DIR, exist_ok=True)


class TempFile:
    """A temporary file that will be deleted on exit.

    This is a context manager that will create a temporary file in the cache
    folder. The file will be deleted on exit.

    Parameters
    ----------
    filename
        The file name to be created. Only the base name will be used.
    parent
        The parent directory where the file will be created.
    delete_on_remove
        If True, the file will be deleted when removed from the manager.

    Examples
    --------
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from astropop.framedata.cache_manager import TempDir, TempFile
    >>> with TemporaryDirectory() as root:
    ...     with TempDir(str(Path(root) / 'owned-cache')) as cache:
    ...         managed = TempFile('message.txt', parent=cache)
    ...         # Registration alone does not create a file; open it to write data.
    ...         print(Path(managed.full_path).exists())
    ...         with managed.open('w') as stream:
    ...             _ = stream.write('example')
    ...         print(Path(managed.full_path).read_text())
    False
    example
    """

    __slots__ = ['filename', 'parent', 'delete_on_remove', 'file']

    def __init__(
        self,
        filename: str,
        parent: TempDir | None=None,
        delete_on_remove: bool=True,
    ) -> None:
        """Register a basename with its parent without opening or creating the file."""
        if os.path.basename(filename) != filename:
            raise ValueError('filename must be a base name, not a full path.')
        # Initialize slots
        self.filename = filename
        if parent is None:
            parent = BaseTempDir
        if isinstance(parent, TempDir):
            self.parent = parent
            if self not in self.parent.managed.values():
                self.parent.managed[self.filename] = self
        else:
            raise ValueError('parent must be None or a TempDir instance.')

        self.delete_on_remove = delete_on_remove

    @property
    def full_path(self) -> str:
        """File path within the managed parent directory."""
        return os.path.join(self.parent.full_path, self.filename)

    @property
    def is_removable(self) -> bool:
        """Whether the current policy permits deleting this file."""
        return self.delete_on_remove

    def delete(self) -> None:
        """Delete this file."""
        if self.is_removable:
            try:
                os.remove(self.full_path)
                if self.parent is not None:
                    del self.parent.managed[self.filename]
            except (FileNotFoundError, KeyError):
                pass
        else:
            logger.warning('File %s marked to be kept.', self.full_path)

    def open(self, mode: str='w') -> _t.IO[_t.Any]:
        """Open the managed path in mode and retain its file object for closing.

        Returns the built-in file object; the default mode truncates existing
        contents. Opening does not change the deletion policy.
        """
        self.file = open(self.full_path, mode)
        return self.file

    def close(self) -> None:
        """Close the retained file object without deleting its path."""
        if self.file is not None:
            self.file.close()
        self.file = None

    def __enter__(self) -> _t.Self:
        """Return this file manager without opening the managed file."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: _types.TracebackType | None,
    ) -> None:
        """Close the retained stream and apply deletion policy on context exit."""
        self.close()
        self.delete()

    def __str__(self) -> str:
        """Return the managed file path."""
        return self.full_path

    def __repr__(self) -> str:
        """Identify the file manager by its full path and object address."""
        return f'<TempFile: "{self.full_path}" at {hex(id(self))}>'
