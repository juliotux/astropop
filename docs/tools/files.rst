.. include:: ../references.txt



Fits File Collections
=====================

Manage and organize FITS files, specially when working with large databases and need to organize files using their header keywords can be a tedious task. The :mod:`astropop.file_collection` module provides utilities to manage and organize FITS files in a database-like fashion.

The basis of this is the |FitsFileGroup| class. It reads FITS files from a folder or a list, and creates a database containing their headers. So, you can easily access the headers and filter files based on their header keywords. This class is also useful to create summaries of the files headers.

This module is mainly designed to work like `~ccdproc.ImageFileCollection`, but its main difference is to work with sqlite databases internally, and the hability to work with persistent headers databases. This may speedup some workflows, specially when working with large databases and compressed files, when headers reading can be very slow.

.. note::

    The |FitsFileGroup| class is designed to only read the files. So, it cannot be used to modify the files.

Initializing a |FitsFileGroup|
------------------------------

The |FitsFileGroup| class is initialized with a list of files, or a folder containing FITS files. If a folder is given, all FITS files in the folder are read. If a list of files is given, only those files are read. The class can also be initialized with a list of files and a folder, in which case, the files in the folder are read and the files in the list are added to the database.

.. ipython::
    :verbatim:

    In [1]: from astropop.file_collection import FitsFileGroup

    # using a folder location
    In [2]: ffg = FitsFileGroup(location='/path/to/data')

    # using a list of files
    In [3]: ffg = FitsFileGroup(files=['/path/to/data/file1.fits',
                                       '/path/to/data/file2.fits'])

Optional keywords also exist to improve the class behavior. The most important are:

* ``ext``: the extension number inside the FITS file to read the header. Default is 0. But if your important data is stored in secondary extensions, you can change this to read the header from there. Like, if your image is stored in the second extension, you can use ``ext=1``.

  .. ipython::
    :verbatim:

    In [4]: ffg = FitsFileGroup(location='/path/to/data', ext=1)

* ``database``: name of the file where the database will be stored in disk. If not given, the database will be stored in memory. If the file already exists, the database will be read from there. If you want to create a new database, you can delete the file before initializing the class.

  .. ipython::
    :verbatim:

    In [5]: ffg = FitsFileGroup(location='/path/to/data', database='files.db')

* ``compression``: if set to True, the reader will also try to find files in compressed format, like ``.fits.gz``. If set to False, which is the default, only uncompressed files will be read.

  .. ipython::
    :verbatim:

    # can also read .fits.gz or .fits.zip files
    In [6]: ffg = FitsFileGroup(location='/path/to/data', compression=True)

* ``glob_include``: If you want to read just some files, you can set them to ``glob_include``, using a `~glob.glob` pattern. For example, if you just want to read files which start with ``BIAS``, you can use ``glob_include='BIAS*'``. All files which match the pattern will be read, the other will be ignored.

  .. ipython::
    :verbatim:

    In [7]: ffg = FitsFileGroup(location='/path/to/data', glob_include='BIAS*')

* ``glob_exclude``: If you want to read all the files, except a few, you can set them to ``glob_exclude``, using a `~glob.glob` pattern. For example, if you want to read all files, except those which start with ``BIAS``, you can use ``glob_exclude='BIAS*'``.

  .. ipython::
    :verbatim:

    In [8]: ffg = FitsFileGroup(location='/path/to/data', glob_exclude='BIAS*')


Files Summary and Header Keyword Values
---------------------------------------

Once the files are read, all headers are stored internally in a database. But a |Table| containing all the headers can be accessed using the `~astropop.file_colletcion.FitsFileGroup.summary` attribute. This table is a copy of the internal database, so modifying it will not affect the database or the filegroup itself.

.. ipython::
    :verbatim:

    In [9]: ffg.summary
    Out[9]:
    <Table length=3>
    FILENAME  EXPTIME  FILTER  OBJECT
    bytes256 float64  bytes8  bytes8
    -------- -------- ------- -------
    file1.fits     1.0     R     star1
    file2.fits     2.0     G     star2
    file3.fits     3.0     R     star3


Also, a full list of the files can be accessed using the `~astropop.file_colletcion.FitsFileGroup.files` attribute.

.. ipython::
    :verbatim:

    In [10]: ffg.files
    Out[10]:
    ['/path/to/data/file1.fits',
     '/path/to/data/file2.fits',
     '/path/to/data/file3.fits']

You can also get a list of the values of a given header keyword using the `~astropop.file_colletcion.FitsFileGroup.values` method. This method returns a list of the values of the given keyword, in the same order as the files in the `~astropop.file_colletcion.FitsFileGroup.files` attribute. If ``unique`` is set to True, only unique values are returned and the order is not guaranteed.

.. ipython::
    :verbatim:

    In [11]: ffg.values('FILTER')
    Out[11]: ['R', 'G', 'R']

    In [12]: ffg.values('FILTER', unique=True)
    Out[12]: ['R', 'G']


Adding or Removing Files
------------------------

Adding or removing files to the group is done using the `~astropop.file_colletcion.FitsFileGroup.add_file` and `~astropop.file_colletcion.FitsFileGroup.remove_files` methods.

To add a file, use `~astropop.file_colletcion.FitsFileGroup.add_file`. Its only argument is ``file`` to set the file name. Prefer using full (absolute) paths for the file name in this function.

.. ipython::
    :verbatim:

    In [13]: ffg.add_file('/path/to/data/file4.fits')

    In [14]: ffg.files
    Out[14]:
    ['/path/to/data/file1.fits',
     '/path/to/data/file2.fits',
     '/path/to/data/file3.fits',
     '/path/to/data/file4.fits']

For remove a file, the `~astropop.file_colletcion.FitsFileGroup.remove_files` accepts a file name with absolute path, or a path relative to the filegroup location. Prefere using absolute paths for the file name in this function too.

.. ipython::
    :verbatim:

    In [15]: ffg.remove_files('/path/to/data/file4.fits')

    In [16]: ffg.files
    Out[16]:
    ['/path/to/data/file1.fits',
     '/path/to/data/file2.fits',
     '/path/to/data/file3.fits']

    In [17]: ffg.remove_files('file1.fits')

    In [18]: ffg.files
    Out[18]:
    ['/path/to/data/file2.fits',
     '/path/to/data/file3.fits']

Adding a Custom Column
----------------------

It is also possible to add a custom column to the database and use it to filter the files. However, as the |FitsFileGroup| is designed to do not change the files, this column/keyword will not be added to the headers in the files. To do this, use the `~astropop.file_colletcion.FitsFileGroup.add_column` method. This method accepts two arguments: ``name`` to set the column name and ``values`` to set the values of the column. The values must be a list with the same length as the number of files in the filegroup.

.. ipython::
    :verbatim:

    In [19]: ffg.add_column('CUSTOM', [1, 2, 3])

    In [20]: ffg.summary
    Out[20]:
    <Table length=3>
    FILENAME  EXPTIME  FILTER  OBJECT  CUSTOM
    bytes256 float64  bytes8  bytes8  int64
    -------- -------- ------- ------- ------
    file1.fits     1.0     R     star1      1
    file2.fits     2.0     G     star2      2
    file3.fits     3.0     R     star3      3

    In [21]: ffg.values('CUSTOM')
    Out[21]: [1, 2, 3]

Filtering and Grouping Files
----------------------------

The main usage of |FitsFileGroup| is to filter, sort and organize FITS files. There are two ways to organize this files: filtering by certaing keyword values or grouping the files by certain keywords. Both return a new |FitsFileGroup| object.

Filtering by Keyword Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method `~astropop.file_colletcion.FitsFileGroup.filtered` receives a dictionary with the keywords and values to filter the files. So, a new |FitsFileGroup| will be created with only the matched files for all the keywords.

.. ipython::
    :verbatim:

    In [22]: ffg_filtered = ffg.filtered({'FILTER': 'R', 'EXPTIME': 1.0})

    In [23]: ffg_filtered.files
    Out[23]: ['/path/to/data/file1.fits']

    In [24]: ffg_filtered.summary
    Out[24]:
    <Table length=1>
    FILENAME  EXPTIME  FILTER  OBJECT  CUSTOM
    bytes256 float64  bytes8  bytes8  int64
    -------- -------- ------- ------- ------
    file1.fits     1.0     R     star1      1

    In [25]: ffg_filtered = ffg.filtered({'FILTER': 'R', 'EXPTIME': 2.0})

    In [26]: ffg_filtered.files
    Out[26]: []

    In [27]: ffg_filtered.summary
    Out[27]: <Table length=0>

Grouping Files
~~~~~~~~~~~~~~

If you want to not only generate a group of files from a single set of keyword valeus, but instead generate multiple groups of files that have the same values in a set of keywords, you can use the `~astropop.file_colletcion.FitsFileGroup.grouped_by` method. This method `yeilds <https://docs.python.org/3/reference/expressions.html#yield-expressions>`_ a new |FitsFileGroup| object for each group of files.

.. Note::

  Since it returns a generator, you must iterate over it to get the groups, like using ``for`` loop.

.. ipython::
  :verbatim:

  In [28]: ffg.summary
  Out[28]:
  <Table length=6>
  FILENAME    EXPTIME  FILTER  OBJECT CUSTOM
  bytes256    float64  bytes8  bytes8  int64
  --------   -------- ------- ------- ------
  file1.fits      1.0    R     star1      1
  file2.fits      2.0    G     star2      1
  file3.fits      3.0    R     star3      1
  file4.fits      1.0    R     star1      2
  file5.fits      2.0    G     star2      2
  file6.fits      3.0    R     star3      2

  In [29]: for group in ffg.grouped_by(['FILTER']):
      ...:     print(f'filter {group.values("FILTER")[0]}')
      ...:     print(f'images {len(group)}')
      ...:     print(group.summary)
      ...:     print('-----------------------------------------')
      ...:
  filter R
  images 4
  <Table length=4>
  FILENAME    EXPTIME  FILTER  OBJECT CUSTOM
  bytes256    float64  bytes8  bytes8  int64
  --------   -------- ------- ------- ------
  file1.fits      1.0    R     star1      1
  file3.fits      3.0    R     star3      1
  file4.fits      1.0    R     star1      2
  file6.fits      3.0    R     star3      2
  -----------------------------------------
  filter G
  images 2
  <Table length=2>
  FILENAME    EXPTIME  FILTER  OBJECT CUSTOM
  bytes256    float64  bytes8  bytes8  int64
  --------   -------- ------- ------- ------
  file2.fits      2.0    G     star2      1
  file5.fits      2.0    G     star2      2
  -----------------------------------------

Iterators
---------

There are also methods for iterating over the files from a |FitsFileGroup|. All these methods are generators that create temporary objects, that are excluded at the end of each loop, so the memory used is just enough to store the current file. To use them, as any `Python generator <https://docs.python.org/3/howto/functional.html#iterators>`_, you can use it inside a ``for`` loop, use the `next` function to get the next file or create a list with them if you want to keep the objects in memory.

- `~astropop.file_collection.FitsFileGroup.hdus`: Iterates over the files getting the selected hdu. Uses `~astropy.io.fits.open` and can accept any argument that `~astropy.io.fits.open` accepts.

  .. ipython::
    :verbatim:

    In [30]: for hdu in ffg.hdus(ext=0):
        ...:     print(hdu)
        ...:
    <astropy.io.fits.hdu.image.PrimaryHDU object at 0xabcdef123456>
    <astropy.io.fits.hdu.image.PrimaryHDU object at 0x654321fedcba>
    <astropy.io.fits.hdu.image.PrimaryHDU object at 0x123456789abc>

- `~astropop.file_collection.FitsFileGroup.data`: Iterates over the files getting the selected hdu and returning the data. Uses `~astropy.io.fits.getdata` and can accept any argument that `~astropy.io.fits.getdata` accepts.

  .. ipython::
    :verbatim:

    In [31]: for data in ffg.data(ext=0):
        ...:     print(data)
        ...:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    [[1 2 3]
     [4 5 6]
     [7 8 9]]

- `~astropop.file_collection.FitsFileGroup.headers`: Iterates over the files getting the selected hdu and returning the header. Uses `~astropy.io.fits.getheader` and can accept any argument that `~astropy.io.fits.getheader` accepts.

  .. ipython::
    :verbatim:

    In [32]: for header in ffg.headers(ext=0):
        ...:     print(header['FILTER'])
        ...:
    R
    G
    R

- `~astropop.file_collection.FitsFileGroup.framedata`: Iterate over the files generating |FrameData| objects from them. Use any argument that `~astropop.framedata.util.read_framedata` method.

  .. ipython::
    :verbatim:

    In [33]: for fd in ffg.framedata():
        ...:     print(fd)
        ...:
    <FrameData object at 0xabcdef123456>
    <FrameData object at 0x654321fedcba>
    <FrameData object at 0x123456789abc>

  .. Note::

    If you want to to create a list of |FrameData| for a large number of files, you may fill all available memory. In this case, use ``use_memmap_backend=True`` that will create temporary `memmap <https://numpy.org/doc/stable/reference/generated/numpy.memmap.html>`_ files to store the data. By default, the files will be created on default system temporary directory. You can change this using the ``cache_folder`` argument.

    .. ipython::
      :verbatim:

      In [34]: ffg.framedata(use_memmap_backend=True, cache_folder='/path/to/my/cache/folder')
      Out[34]:
      [<FrameData object at 0xabcdef123456>,
       <FrameData object at 0x654321fedcba>,
       <FrameData object at 0x123456789abc>]

File Collection API
-------------------

.. automodapi:: astropop.file_collection
    :no-inheritance-diagram:
