Tutorials
=========

Choose one of three observing workflows: a CCD stack, an individual-exposure
light curve, or dual-beam polarimetry. The nine notebook companions share an
archive, but do not all process the same observations. Each notebook is generated
from its tutorial's numerical examples and includes printed results and figures.

.. plot::
    :include-source: false
    :alt: Bias and flat masters feed the science stack and a separate time series. Stack photometry leads to astrometric and photometric calibration. Paired beam measurements feed polarimetry independently.
    :caption: Follow the arrows for data dependencies. The light curve uses a separate twelve-exposure sequence; polarimetry starts from paired-beam tables, independent of the imaging stack.

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 4.8), layout='constrained')
    nodes = {
        'bias': (0, 3.2, 'Master\nbias', 'C0'),
        'flat': (2, 3.2, 'Master\nflat', 'C0'),
        'stack': (4, 3.2, 'Calibrate +\nregister + stack', 'C0'),
        'measure': (6, 3.2, 'Stack\nphotometry', 'C0'),
        'wcs': (8, 3.2, 'Fit\nastrometry', 'C0'),
        'magnitude': (10, 3.2, 'Calibrate\nmagnitudes', 'C0'),
        'series': (4, 1.6, 'Calibrate separate\ntime-series images', 'C1'),
        'curve': (6, 1.6, 'Differential\nlight curve', 'C1'),
        'beams': (4, 0, 'Paired beam\nmeasurements', 'C2'),
        'stokes': (6, 0, 'Fit Stokes\nparameters', 'C2'),
    }
    for x, y, label, color in nodes.values():
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                bbox={'boxstyle': 'round,pad=0.45', 'fc': 'white', 'ec': color})
    edges = [('bias', 'flat'), ('flat', 'stack'), ('stack', 'measure'),
             ('measure', 'wcs'), ('wcs', 'magnitude'), ('flat', 'series'),
             ('series', 'curve'), ('beams', 'stokes')]
    for start, end in edges:
        ax.annotate('', xy=nodes[end][:2], xytext=nodes[start][:2],
                    arrowprops={'arrowstyle': '->', 'shrinkA': 50, 'shrinkB': 50,
                                'color': '0.4'})
    ax.annotate('', xy=nodes['stack'][:2], xytext=nodes['bias'][:2],
                arrowprops={'arrowstyle': '->', 'shrinkA': 34, 'shrinkB': 48,
                            'connectionstyle': 'arc3,rad=-0.35', 'color': '0.4'})
    ax.text(0, 1.6, 'Separate raw sequence;\nsame bias and flat masters',
            va='center', fontsize=10, color='C1')
    ax.text(0, 0, 'Separate polarimetric data;\nno input from the imaging stack',
            va='center', fontsize=10, color='C2')
    ax.set(xlim=(-1, 11), ylim=(-0.6, 4.2))
    ax.axis('off')

Download and run the examples
-----------------------------

.. tutorial-data::

Start with :doc:`../installation` to select a Python environment and Jupyter
kernel. Extract the archive into a working directory and keep this layout:

.. code-block:: text

    data/raw/               bias, flat and science exposures for the stack
    data/time_series/raw/   separate variable-star exposures
    data/polarimetry.ecsv    ordinary/extraordinary beam measurements
    data/polarimetry_quarterwave.ecsv   noisy quarter-wave beam measurements
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
