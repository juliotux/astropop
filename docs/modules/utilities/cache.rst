Temporary storage and cache ownership
=====================================

FrameData can store arrays in memory-mapped files, and ImCombiner can use a
disk cache for its private input copies. Prefer those interfaces for reduction
work. :mod:`astropop.framedata.cache_manager` is the lower-level manager for
owned temporary files and directories; it is not a persistent image database.

Managed paths
-------------

``BaseTempDir`` is the shared manager rooted at ``CACHE_DIR``, beneath Astropy's
cache directory. A ``TempDir`` with a relative name is placed beneath its
parent (or BaseTempDir). An absolute name creates a managed top-level directory.
A ``TempFile`` name must be a basename; use a parent directory to choose its
location. Creating a TempFile manager does not open or populate the file.

Use a dedicated directory, never an existing directory containing observations
or unrelated files: deleting a removable TempDir recursively removes its
contents, including unmanaged files. The context managers request deletion on
exit, including exception paths::

    >>> from tempfile import TemporaryDirectory
    >>> from pathlib import Path
    >>> from astropop.framedata.cache_manager import TempDir

    >>> with TemporaryDirectory() as root:
    ...     with TempDir(str(Path(root) / 'owned-cache')) as directory:
    ...         managed = directory.create_file('message.txt')
    ...         print('File manager:', type(managed).__name__)
    ...         with managed.open('w') as stream:
    ...             _ = stream.write('example')
    ...         print('Path(managed.full_path).read_text():', Path(managed.full_path).read_text())
    ...     print('Path(directory.full_path).exists():', Path(directory.full_path).exists())
    File manager: TempFile
    Path(managed.full_path).read_text(): example
    Path(directory.full_path).exists(): False

Deletion policy
---------------

``delete_on_remove=True`` permits deletion. Setting it to ``False`` keeps that
entry and prevents deletion of a containing directory that still owns it.
``is_removable`` reports whether deletion is currently allowed. ``delete()``
requests deletion according to that policy; it does not override retained
children. Close open files and release memory mappings before deleting them,
especially on Windows.

The parent checks its managed children before deletion. This example retains
one file, observes the resulting policy, then permits explicit cleanup::

    >>> with TemporaryDirectory() as root:
    ...     directory = TempDir(str(Path(root) / 'owned-cache'))
    ...     retained = directory.create_file('keep.txt', delete_on_remove=False)
    ...     with retained.open('w') as stream:
    ...         _ = stream.write('keep')
    ...     directory.delete()
    ...     print('Parent removable:', directory.is_removable)
    ...     print('File retained:', Path(retained.full_path).exists())
    ...
    ...     # Release the child's retention policy so its parent can also be deleted.
    ...     retained.delete_on_remove = True
    ...     directory.delete()
    ...     print('Parent deleted:', not Path(directory.full_path).exists())
    Parent removable: False
    File retained: True
    Parent deleted: True

At interpreter shutdown, the registered ``cleanup`` function removes eligible
managed directories if the module variable ``DELETE_ON_EXIT`` is true. Change
that variable on the module itself if needed; assigning an independently
imported Boolean name does not update the module setting. Shutdown cleanup is
a fallback, not a replacement for explicit ownership. Do not manually invoke
all Python exit handlers to clean an individual cache.

FrameData's disabled-storage settings prevent optional arrays from being
allocated only when set before frames are loaded. Hiding an existing mapped
array does not delete its file; see :ref:`optional-frame-arrays`.




Reference/API
-------------

.. automodapi:: astropop.framedata.cache_manager
    :no-inheritance-diagram:

.. currentmodule:: None

.. py:data:: astropop.framedata.cache_manager.BaseTempDir

    Shared TempDir instance at ``CACHE_DIR``. Relative top-level manager names
    are created beneath this instance.

.. currentmodule:: None

.. py:data:: astropop.framedata.cache_manager.CACHE_DIR

    Path beneath Astropy's cache directory used for Astropop-managed files.
    The value is established when the module is imported.

.. currentmodule:: None

.. py:data:: astropop.framedata.cache_manager.DELETE_ON_EXIT

    Module-level Boolean, initially ``True``, controlling shutdown cleanup.
    Individual entries' retention policies still apply when cleanup is enabled.


.. currentmodule:: None
