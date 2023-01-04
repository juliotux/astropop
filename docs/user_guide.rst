.. include:: references.txt

User Guide
==========

|astropop| is a collection of several routines to reduce astronomical data, focused on polarimetry, photometry and astrometry data. It implements all (or almost all) what is required to perform these reductions.

So, |astropop| is not a instrument specific pipeline and, instead, it is a library, intended to make it easier to you create your own reduction scripts and pipelines.

|astropop| has several modules. For and astronomer reducing data, the interesting documentation for you is described here.

Data structues
--------------

.. toctree::
    :maxdepth: 1

    framedata.rst
    memmap.rst
    physical.rst

Basic reduction routines
------------------------

.. toctree::
    :maxdepth: 1

    imarith.rst
    ccdprocessing.rst
    photometry.rst
    registering.rst
    polarimetry.rst
    astrometry.rst
    pipelines.rst

General utilities
-----------------

.. toctree::
    :maxdepth: 1

    files.rst
    logger.rst
    testing.rst
    py_utils.rst
    fits_utils.rst

Online catalogs
---------------

.. toctree::
    :maxdepth: 1

    catalogs.rst
