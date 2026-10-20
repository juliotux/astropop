Tutorials
=========

Work through a complete observing sequence in nine stages. Each page provides
a notebook download generated from its numerical examples, along with printed
output and relevant diagnostic figures. The shared archive contains all nine
notebooks, synthetic inputs, local catalogs and reference intermediate products.
Data creation is hidden from the reduction walkthroughs; no live catalog query
is needed.

.. tutorial-data::

Run the notebooks in order to use your own preceding products in ``reduced/``.
For an independent stage, the examples fall back to ``data/reference`` when its
required input product is absent. Keep the extracted archive layout unchanged
so notebooks can locate their helpers and data. A notebook's hidden setup cell
sets optional-array configuration before loading frames; the visible cells
perform the actual reduction and measurement steps.

.. toctree::
    :maxdepth: 1

    observations
    master_bias
    master_flat
    science_reduction
    photometry
    light_curve
    polarimetry
    astrometric_calibration
    photometric_calibration

Start with :doc:`../installation` for environment setup.
:doc:`ccd_reduction` summarizes the CCD stages and their assumptions.
The light-curve workflow uses individual exposures, rather than the stack.
Polarimetry uses separate beam fluxes and retarder angles. Astrometry starts
from approximate pointing and local counterparts; it does not execute a blind
external solver.

Additional walkthroughs
------------------------

.. toctree::
    :maxdepth: 1

    ccd_reduction
    local_catalog_matching

:doc:`local_catalog_matching` explains search radii and unmatched positions
before interpreting associations as astrometric or photometric calibration.
