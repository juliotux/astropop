Tutorials
=========

Choose one of three observing workflows: a CCD stack, an individual-exposure
light curve, or dual-beam polarimetry. The nine notebook companions share an
archive, but do not all process the same observations. Each notebook is generated
from its tutorial's numerical examples and includes printed results and figures.

.. plot:: _ext/task_figures.py plot_tutorial_paths
    :include-source: False
    :alt: Bias and flat masters feed the science stack and a separate time series. Stack photometry leads to astrometric and photometric calibration. Paired beam measurements feed polarimetry independently.
    :caption: Follow the arrows for data dependencies. The light curve uses a separate twelve-exposure sequence; polarimetry starts from an eight-position beam table.

Download and run the examples
-----------------------------

.. tutorial-data::

Start with :doc:`../installation` to select a Python environment and Jupyter
kernel. Extract the archive into a working directory and keep this layout:

.. code-block:: text

    data/raw/               bias, flat and science exposures for the stack
    data/time_series/raw/   separate variable-star exposures
    data/polarimetry.ecsv    ordinary/extraordinary beam measurements
    data/reference/         products for independent stages
    notebooks/              nine runnable tutorial notebooks
    *.py                    local plotting and data helpers

The notebook setup locates ``data`` from the bundle. When copying examples from
the HTML pages, run from the extracted bundle root so ``Path('data')`` resolves
correctly. Keep uncertainties and flags enabled **before loading frames**:

.. code-block:: python

    from astropop.config import AstropopConfig

    AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = False
    AstropopConfig.FRAMEDATA_DISABLE_FLAGS = False

The notebooks apply this policy in their setup cell. It is also needed for
visible HTML examples; see :doc:`../concepts/configuration` if you changed
configuration earlier in the interpreter.

Derived products are written to ``reduced/`` and overwritten on reruns. Stages
can use complete input sets from ``data/reference`` to run independently.
Calibration tutorials reject partial local product sets instead of mixing them
with reference files. After changing a crop, registration reference or measurement
settings, rerun downstream stages together; file existence does not prove freshness.

CCD stack and calibration
--------------------------

.. toctree::
    :maxdepth: 1

    observations
    master_bias
    master_flat
    science_reduction
    photometry
    astrometric_calibration
    photometric_calibration

:doc:`ccd_reduction` summarizes the detector assumptions and the CCD stages.
Astrometry starts from approximate pointing and local counterparts; it does not
execute a blind external solver.

Separate observing sequences
----------------------------

.. toctree::
    :maxdepth: 1

    light_curve
    polarimetry

The light curve uses individual exposures with the same detector calibration.
Polarimetry uses paired beam fluxes and retarder angles rather than the CCD stack.

Additional walkthroughs
------------------------

.. toctree::
    :maxdepth: 1

    ccd_reduction
    local_catalog_matching
    astrometry_net
    online_catalogs

:doc:`local_catalog_matching` explains search radii and unmatched positions
before interpreting associations as astrometric or photometric calibration.
The Astrometry.net and online-catalog tutorials require your own image/index files or
live services, respectively; their external recipes are not run by offline
doctests or included in the nine generated notebook companions.
