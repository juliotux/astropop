.. include:: ../references.txt

Fits File Collections
=====================

|FitsFileGroup| indexes FITS headers in a SQLite database so you can select
observations without loading every image. Start with
:doc:`../how_to/file_collections` for an executable example; this page explains
index ownership, selection and persistence. Editing a group changes its index,
not the FITS files on disk.

Initializing a |FitsFileGroup|
------------------------------

Supply ``location`` to scan a directory recursively, or ``files`` for an explicit
list. The arguments are mutually exclusive. ``ext`` selects the header extension;
an empty primary HDU can require ``ext=1`` or an extension name.

.. code-block:: python

    from pathlib import Path
    from astropop.file_collection import FitsFileGroup, list_fits_files

    folder = Path('/path/to/data').resolve()
    group = FitsFileGroup(location=folder, ext=1, database='headers.db')

``database`` defaults to an in-memory index. A path creates a persistent database
or reopens an existing one; reopening does not automatically refresh headers.
Call ``group.update(location=folder)`` to rebuild after changes. Rebuilding can
remove custom index columns, so keep their provenance separately.

``compression=True`` includes recognized compressed FITS suffixes in directory
scans. ``fits_ext`` overrides the searched suffixes. ``glob_include`` and
``glob_exclude`` use shell-style patterns against the discovered path, including
its directory. For example, ``str(folder / 'BIAS*')`` matches paths beginning
with that absolute prefix; ``'BIAS*'`` alone does not match absolute paths.
To select basenames across subdirectories, build the list explicitly:

.. code-block:: python

    bias_files = [name for name in list_fits_files(folder)
                  if Path(name).name.startswith('BIAS')]
    biases = FitsFileGroup(files=bias_files, ext=1)
    print('Selected bias files:', len(biases))

Files Summary and Header Keyword Values
---------------------------------------

``group.summary`` returns a Table copy. Modifying that copy does not update the
index. ``group.files`` returns resolved file paths. Persistent databases store
paths relative to the database directory, so moving the database alone can
break its relationship to the observations.

.. code-block:: python

    print('Indexed files:', len(group))
    print('Columns:', group.summary.colnames)
    print('Filters:', group.values('FILTER', unique=True))

``values`` preserves file order unless ``unique=True`` is requested. Header
columns are normalized in the summary; inspect ``summary.colnames`` before
using names in operations that require exact spelling.

Adding or Removing Files
------------------------

``add_file`` reads a file's header into the group. ``remove_file`` removes its
entry from the index without deleting the observation. Use resolved paths when
updating a group to avoid ambiguity about relative filenames.

.. code-block:: python

    new_file = folder / 'new_exposure.fits'
    group.add_file(new_file)
    print('Indexed after addition:', len(group))
    group.remove_file(new_file)
    print('Indexed after removal:', len(group))

Adding a Custom Column
----------------------

Store a per-file classification in the index with ``add_column``. Values must
correspond to the current file order. The column is not written to FITS headers.

.. code-block:: python

    group.add_column('reviewed', [False] * len(group))
    print('Review state:', group.values('reviewed'))

Filtering and Grouping Files
----------------------------

Filtering and grouping return new groups. They select references to the existing
observations; they do not copy or calibrate the underlying FITS data.

Filtering by Keyword Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``filtered`` requires every supplied keyword/value criterion to match. Keyword
lookup here is case-insensitive. Check the selection count before combining it.

.. code-block:: python

    red_science = group.filtered({'FILTER': 'R', 'IMAGETYP': 'LIGHT'})
    print('R-band science exposures:', len(red_science))

Grouping Files
^^^^^^^^^^^^^^

``group_by`` yields groups sharing the selected header values. Unlike filtering,
it requires exact summary-column names. For a lowercase ``filter`` column:

.. code-block:: python

    for subset in group.group_by(['filter']):
        print('Filter:', subset.values('FILTER', unique=True),
              'frames:', len(subset))

For calibration selection, group by detector, binning, readout mode and geometry
as well as filter where appropriate. Those keyword names are instrument-specific;
see :doc:`../reduction/pipelines` for an application-level policy.

Iterators
---------

The iterators load files on demand. Choose the data extension explicitly when
iterating, even if you indexed another extension's header:

.. code-block:: python

    import numpy as np

    for frame in red_science.framedata(ext=1):
        print('Shape:', frame.shape, 'median:', float(np.median(frame.data)))

``headers`` yields headers, ``data`` yields image arrays, and ``hdus`` yields
image HDUs. ``framedata`` converts the selected HDU to |FrameData|. Keep only
measurements or output paths when processing a large group: converting an
iterator to a list retains all its objects, and the last loop variable also
remains bound after iteration.

``use_memmap_backend=True`` on ``framedata`` moves its arrays into temporary
mapped files. ``cache_folder`` controls their location; this does not impose a
whole-process RAM limit. See :doc:`../data_structures/cache` for cache ownership
and cleanup, and :ref:`imcombine-memory-planning` for combination storage.

File Collection API
-------------------

See :doc:`../reference/utilities` for signatures and parameter contracts.

.. raw:: html

    <span id="astropop-file-collection-module"></span>
    <span id="classes"></span>
    <span id="functions"></span>
    <span id="module-astropop.file_collection"></span>
