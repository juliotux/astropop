.. include:: ../references.txt

Fits File Collections
=====================

Manage and organize FITS files, specially when working with large databases and need to organize files using their header keywords can be a tedious task. The :mod:`astropop.file_collection` module provides utilities to manage and organize FITS files in a database-like fashion.

The basis of this is the |FitsFileGroup| class. It reads FITS files from a folder or a list, and creates a database containing their headers. So, you can easily access the headers and filter files based on their header keywords. This class is also useful to create summaries of the files headers.

This module is mainly designed to work like :class:`ccdproc.ImageFileCollection`, but its main difference is to work with sqlite databases internally, and the hability to work with persistent headers databases. This may speedup some workflows, specially when working with large databases and compressed files, when headers reading can be very slow.

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


Files Summary
-------------

Once the files are read, all headers are stored internally in a database. But a |Table| containing all the headers can be accessed using the :attr:`FitsFileGroup.summary` attribute. This table is a copy of the internal database, so modifying it will not affect the database or the filegroup itself.


Adding or Removing Files
------------------------


Filtering Files
---------------


Iterators
---------



File Collections API
--------------------

.. automodapi:: astropop.file_collection
    :no-inheritance-diagram:
