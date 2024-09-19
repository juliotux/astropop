# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Managing the cache folder for FrameData memmaping."""

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


def cleanup():
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
    dirname : str
        The directory name to be created.
    delete_on_remove : bool, optional
        If True, the directory will be deleted when removed from the manager.
    is_full_path : bool, optional
        If True, the dirname is already a full path. Else, a cache folder will
        be prepended.
    """

    __slots__ = ['dirname', 'parent', 'managed', 'delete_on_remove']

    def __init__(self, dirname, parent=None, delete_on_remove=True):
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
    def full_path(self):
        if self.parent is None:
            return self.dirname
        return os.path.join(self.parent.full_path, self.dirname)

    def create_file(self, basename, delete_on_remove=True):
        """Add a TempFile to this folder.

        Parameters
        ----------
        basename: str
            Base name of the file to be created.
        delete_on_remove: bool
            Keep the file after the program exits.

        Returns
        -------
        TempFile:
            The created TempFile instance.
        """
        f = TempFile(basename, parent=self,
                     delete_on_remove=delete_on_remove)
        self.managed[basename] = f
        return f

    def create_folder(self, basename, delete_on_remove=True):
        """Add a TempFolder to this folder.

        Parameters
        ----------
        basename: str
            Base name of the file or folder to be created.
        delete_on_remove: bool
            Keep the file after the program exits.

        Returns
        -------
        TempDir:
            The created TempDir instance.
        """
        f = TempDir(basename, parent=self,
                    delete_on_remove=delete_on_remove)
        self.managed[basename] = f
        return f

    def delete(self):
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
    def is_removable(self):
        """This folder can be removed and all its contents."""
        keep = not self.delete_on_remove
        for i in self.managed.values():
            keep = keep or not i.is_removable
        return not keep

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.delete()

    def __str__(self):
        return self.dirname

    def __repr__(self):
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
    filename : str
        The file name to be created. Only the base name will be used.
    parent : TempDir
        The parent directory where the file will be created.
    delete_on_remove : bool, optional
        If True, the file will be deleted when removed from the manager.
    """

    __slots__ = ['filename', 'parent', 'delete_on_remove', 'file']

    def __init__(self, filename, parent=None, delete_on_remove=True):
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
    def full_path(self):
        return os.path.join(self.parent.full_path, self.filename)

    @property
    def is_removable(self):
        return self.delete_on_remove

    def delete(self):
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

    def open(self, mode='w'):
        self.file = open(self.full_path, mode)
        return self.file

    def close(self):
        if self.file is not None:
            self.file.close()
        self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        self.delete()

    def __str__(self):
        return self.full_path

    def __repr__(self):
        return f'<TempFile: "{self.full_path}" at {hex(id(self))}>'
