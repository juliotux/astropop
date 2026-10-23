.. include:: ../../references.txt

Fits File Collections
=====================

|FitsFileGroup| indexes FITS headers in a SQLite database so you can select
observations without loading every image. This chapter explains the constructor,
summary properties, filtering/grouping methods and image iterators, with simple
examples for each interface. For a complete extension-selection and database
reuse workflow, see :doc:`../../tutorials/diagnostics/fits_collections`. Editing a group changes its index,
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

    # Index the science extension rather than the empty primary HDU.
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

Inspect a small explicit group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this example, ``example_folder`` contains ``image_0.fits`` through
``image_2.fits``. Each has an empty primary HDU and a ``SCI`` image extension
with a ``FILTER`` header. Listing paths first lets an application inspect or
modify the input list before constructing its group:

.. doctest::
    :hide:

    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> import numpy as np
    >>> from astropy.io import fits

    >>> _file_example = TemporaryDirectory()
    >>> example_folder = Path(_file_example.name).resolve()
    >>> for index, band in enumerate(('R', 'B', 'R')):
    ...     image = fits.ImageHDU(np.full((2, 3), index + 1.), name='SCI')
    ...     image.header['FILTER'] = band
    ...     image.header['BUNIT'] = 'adu'
    ...     fits.HDUList([fits.PrimaryHDU(), image]).writeto(example_folder / f'image_{index}.fits')

.. doctest::

    >>> from astropop.file_collection import FitsFileGroup, list_fits_files

    >>> paths = list_fits_files(example_folder)
    >>> small = FitsFileGroup(files=paths, ext='SCI', database=str(example_folder / 'example.db'))

    >>> print('Files:', [Path(name).name for name in small.files])
    Files: ['image_0.fits', 'image_1.fits', 'image_2.fits']
    >>> print('Filter column:', [key for key in small.keys if key == 'filter'])
    Filter column: ['filter']
    >>> print('Filter values:', small['filter'])
    Filter values: ['R', 'B', 'R']
    >>> print('First-file group:', len(small[0]), '; first-two group:', len(small[:2]))
    First-file group: 1 ; first-two group: 2

    >>> stored = small.relative_path(small.files[0])

    >>> print('Stored path:', stored)
    Stored path: image_0.fits
    >>> print('Resolved file:', Path(small.full_path(stored)).name)
    Resolved file: image_0.fits

``keys`` lists indexed column names. String indexing selects a column using
its actual name; integer indexing returns a one-file group, and a slice returns
a group with the selected rows. These groups refer to the original observations;
they do not write new image files. ``relative_path`` and ``full_path`` translate
between database-relative filenames and filesystem paths. In-memory databases
have no database directory, so there is no relative-path conversion in that case.

The alternate iterators below read the same selected ``SCI`` extension but
return different objects. This example retains only short summaries::

    >>> print('Header filters:', [header['FILTER'] for header in small.headers(ext='SCI')])
    Header filters: ['R', 'B', 'R']
    >>> print('Image means:', [float(np.mean(pixels)) for pixels in small.data(ext='SCI', memmap=False)])
    Image means: [1.0, 2.0, 3.0]
    >>> print('HDU names:', [hdu.name for hdu in small.hdus(ext='SCI', memmap=False)])
    HDU names: ['SCI', 'SCI', 'SCI']
    >>> print('Frame units:', [str(frame.unit) for frame in small.framedata(ext='SCI')])
    Frame units: ['adu', 'adu', 'adu']

Use ``headers`` when pixel data are unnecessary, ``data`` for array-only
processing, ``hdus`` for both header and data, and ``framedata`` when reduction
functions need units and optional arrays. These are read interfaces; changing
a yielded object does not automatically write the source FITS file.

Release the group before removing its database file so its SQLite connection
is closed, including on Windows::

    >>> del small

.. doctest::
    :hide:

    >>> _file_example.cleanup()

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

    # Select matching header values before opening the pixel arrays.
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
see :doc:`../../tutorials/ccd_reduction/policies` for an application-level policy.

Iterators
---------

The iterators load files on demand. Choose the data extension explicitly when
iterating, even if you indexed another extension's header:

.. code-block:: python

    import numpy as np

    # Iterate over frames instead of keeping every image array in a list.
    for frame in red_science.framedata(ext=1):
        print('Shape:', frame.shape, 'median:', float(np.median(frame.data)))

``headers`` yields headers, ``data`` yields image arrays, and ``hdus`` yields
image HDUs. ``framedata`` converts the selected HDU to |FrameData|. Keep only
measurements or output paths when processing a large group: converting an
iterator to a list retains all its objects, and the last loop variable also
remains bound after iteration.

``use_memmap_backend=True`` on ``framedata`` moves its arrays into temporary
mapped files. ``cache_folder`` controls their location; this does not impose a
whole-process RAM limit. See :doc:`../utilities/cache` for cache ownership
and cleanup, and :ref:`imcombine-memory-planning` for combination storage.

File Collection API
-------------------

See :doc:`../utilities/index` for signatures and parameter contracts.



Input and result diagnostics
----------------------------

.. list-table:: Symptoms, checks and recovery
    :header-rows: 1
    :widths: 24 40 36

    * - Symptom
      - Minimal diagnostic
      - Recovery and contract
    * - Grouping FITS files says a key column is missing
      - Inspect ``group.summary.colnames``.
      - Use exact case for ``group_by`` even if header filtering accepted
        uppercase names. See :doc:`../../tutorials/diagnostics/fits_collections`.


Reference/API
-------------

.. automodapi:: astropop.file_collection
    :no-inheritance-diagram:


.. currentmodule:: None

Related topics
--------------

.. toctree::
    :maxdepth: 1

    fits_utils
