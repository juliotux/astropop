Astropop documentation
======================

.. py:module:: astropop

Astropop is a Python library for astronomical image reduction, with tools for
photometry, polarimetry, astrometry, and catalog matching. Combine its functions
in scripts tailored to your detector and instrument; the library does not choose
a universal reduction recipe.

Getting started
---------------

.. toctree::
    :maxdepth: 1

    installation
    user_guide

The user guide describes the current library interfaces and their numerical,
unit, mask, and uncertainty conventions. Start with FrameData, then choose the
reduction and utility chapters needed for your workflow.

Worked notebooks
----------------

These notebooks illustrate complete workflows. They are stored examples and
are not executed during documentation builds. Adapt filenames, external
services, and instrument parameters to your observations.

.. toctree::
    :maxdepth: 1

    ipynb/diy_reduction_script
    ipynb/astrometry_example
    ipynb/online_catalogs

Developer documentation
-----------------------

.. toctree::
    :maxdepth: 1

    dev/pixel_convention
    dev/testing
    dev/internal_interfaces

API reference
-------------

Each guide ends with the corresponding module's API reference. The
:ref:`modindex` and :ref:`genindex` provide entry points by module and symbol.
