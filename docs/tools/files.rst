.. include:: ../references.txt

Fits File Collections
=====================

Manage and organize FITS files, specially when working with large databases and need to organize files using their header keywords can be a tedious task. The :mod:`astropop.file_collection` module provides utilities to manage and organize FITS files in a database-like fashion.

The basis of this is the |FitsFileGroup| class. It reads FITS files from a folder or a list, and creates a database containing their headers. So, you can easily access the headers and filter files based on their header keywords. This class is also useful to create summaries of the files headers.

This module is mainly designed to work like :class:`ccdproc.ImageFileCollection`, but its main difference is to work with sqlite databases internally, and the hability to work with persistent headers databases. This may speedup some workflows, specially when working with large databases and compressed files, when headers reading can be very slow.

Initializing a |FitsFileGroup|
------------------------------




File Collections API
--------------------

.. automodapi:: astropop.file_collection
    :no-inheritance-diagram:
