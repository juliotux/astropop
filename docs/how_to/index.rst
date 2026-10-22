Task guides
===========

These chapters document Astropop's modules: what their functions and classes
compute, how to use them, and how to interpret their results. Each chapter
combines simple examples with the algorithms, units and assumptions relevant
to that API. Exact signatures are in :doc:`../reference/index`.

For a complete reduction involving several modules, follow
:doc:`../tutorials/index`. Shared reduction policies and troubleshooting are
in :doc:`../concepts/index`.

Image containers and quantities
-------------------------------

.. toctree::
    :maxdepth: 1

    ../data_structures/framedata
    ../data_structures/physical
    ../data_structures/cache
    ../tools/flags

Image processing
----------------

.. toctree::
    :maxdepth: 1

    ../reduction/ccdprocessing
    image_arithmetic
    image_combination
    ../reduction/registering

Measurements and calibration
----------------------------

.. toctree::
    :maxdepth: 1

    ../reduction/photometry
    ../reduction/polarimetry
    ../reduction/astrometry

Catalogs
--------

.. toctree::
    :maxdepth: 1

    ../tools/catalogs

Files and numerical utilities
-----------------------------

.. toctree::
    :maxdepth: 1

    ../tools/files
    ../tools/fits_utils
    ../tools/math
    ../tools/logger
    ../tools/py_utils
