Download and run the examples
-----------------------------

.. tutorial-data::

Start with :doc:`installation` to select a Python environment and Jupyter
kernel. Extract the archive into a working directory and keep this layout:

.. code-block:: text

    data/raw/               bias, flat and science exposures for the stack
    data/time_series/raw/   separate variable-star exposures
    data/polarimetry/images/ ideal calibrated half-wave beam images
    data/polarimetry.ecsv    injected half-wave beam fluxes for reference
    data/polarimetry_quarterwave.ecsv   noisy quarter-wave beam measurements
    data/reference/         products for independent stages
    notebooks/              ten runnable tutorial notebooks
    *.py                    local plotting and data helpers

The notebook setup locates ``data`` from the bundle. When copying examples from
the HTML pages, run from the extracted bundle root so ``Path('data')`` resolves
correctly. Keep uncertainties and flags enabled **before loading frames**:

.. code-block:: python

    from astropop.config import AstropopConfig

    # Enable these arrays before reading frames so the examples can propagate them.
    AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = False
    AstropopConfig.FRAMEDATA_DISABLE_FLAGS = False

The notebooks apply this policy in their setup cell. It is also needed for
visible HTML examples; see :doc:`../modules/utilities/configuration` if you changed
configuration earlier in the interpreter.

Derived products are written to ``reduced/`` and overwritten on reruns. Stages
can use complete input sets from ``data/reference`` to run independently.
Calibration tutorials reject partial local product sets instead of mixing them
with reference files. After changing a crop, registration reference or measurement
settings, rerun downstream stages together; file existence does not prove freshness.
