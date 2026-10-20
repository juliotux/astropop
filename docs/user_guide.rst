.. include:: references.txt

User Guide
==========

For a learning sequence use :doc:`tutorials/index`; for a particular task use
:doc:`how_to/index`. Shared assumptions are in :doc:`concepts/index`, and exact
callable contracts are in :doc:`reference/index`. This page retains the original
module-based navigation.

Astropop is a library for building astronomical reduction scripts. It provides
image containers, physical quantities, calibration and combination, registration,
photometry, polarimetry, astrometry, catalog matching, and file utilities.
Your application chooses the instrument-specific recipe and calibration inputs.

Start with the data structures to understand units, uncertainties, flags, and
storage. The reduction chapters explain the numerical operations; the utilities
chapters cover file organization, catalogs, and supporting helpers. Installation
and a short first example are in :doc:`installation`.

.. toctree::
    :maxdepth: 2

    data_structures.rst
    reducing.rst
    tools.rst
