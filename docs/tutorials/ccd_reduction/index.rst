Reduce a small CCD observing sequence
=====================================

Follow a reduction from raw observations to instrumental measurements using
shared synthetic FITS files. Each stage has its own page and executable notebook;
the numerical steps are maintained in the pages and generate the notebook
companions. Start at the beginning or use the supplied reference products to
run a later stage independently.

Get the observations
---------------------

.. tutorial-data::

Extract the archive once. It includes ``data/``, ten numbered ``notebooks/``,
and the source helpers needed to reproduce the observations offline.
``data/raw`` contains eight bias frames, six flats and six science exposures.
``data/time_series/raw`` holds a separate twelve-exposure variable-star sequence.
``data/reference`` contains intermediate products for standalone workflows.

The simulated detector has 256 by 256 pixels, a gain of one electron per ADU,
4-electron read noise, a structured bias and a sensitivity gradient with a dust
shadow. Science exposures contain nine isolated stars, a sloping sky, Poisson
noise, small pointing shifts and injected bright edge events. Flat illumination
varies between exposures. Dark current, overscan, nonlinearity and saturation
are not simulated. These are teaching data, not a measured model of a camera.

Start with :doc:`observations` to inspect HDUs, headers and image geometry,
then save an observing log and selected file lists.

Build the master bias and normalized flat
------------------------------------------

:doc:`master_bias` constructs the additive correction. :doc:`master_flat`
subtracts it from each flat before normalizing and combining the response.
Both stages save FITS products in ``reduced/``. Keep uncertainties and flags
enabled before loading frames for these examples; see
:doc:`../../modules/utilities/configuration` for optional storage.

Calibrate individual science frames
-----------------------------------

:doc:`science_reduction` reads those master products, estimates raw-frame
uncertainties, applies calibration and saves individual corrected exposures.
It explains the gain convention and the detector-quality policy explicitly.
Do not substitute a scalar bias or a raw flat for those master corrections.

Register and reject isolated events
-----------------------------------

The same :doc:`science_reduction` workflow estimates translations from a central
window, applies them to the full images and combines the aligned exposures with
clipping. Review :doc:`../diagnostics/registration_diagnostics` for center/edge
residuals, and :doc:`../../modules/image/imcombine` for estimator and rejection
choices. Registration, outlier rejection and distortion correction are distinct.

Save and inspect the products
------------------------------

Follow :doc:`../photometry/photometry` for the stack, or :doc:`../photometry/light_curve` for measurements on
individual exposures. :doc:`../astrometry_catalogs/astrometric_calibration` attaches a fitted WCS using
local standards; :doc:`../photometry/photometric_calibration` then converts instrumental
fluxes to a stated magnitude convention. :doc:`../polarimetry/halfwave` uses a separate
paired-beam input rather than treating ordinary images as polarimetric data.

Inspect figures and printed output at each stage. The examples overwrite named
derived products so notebooks can be rerun; raw files remain inputs. Source
truth is used only for recovery diagnostics. A small residual in the simulation
does not establish accuracy for different detectors or observations.

.. plot::
    :include-source: false
    :alt: Bias and flat masters feed the science stack and a separate time series. Stack photometry leads to astrometric and photometric calibration. Paired beam measurements feed polarimetry independently.
    :caption: Follow the arrows for data dependencies. The light curve uses a separate twelve-exposure sequence; half-wave polarimetry starts from paired-beam images and quarter-wave polarimetry from a flux table, both independent of the imaging stack.

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
        'beams': (4, 0, 'Detect + measure\npaired beams', 'C2'),
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


Follow the CCD reduction
------------------------

.. toctree::
    :maxdepth: 1

    observations
    master_bias
    dark_correction
    master_flat
    science_reduction
    policies

The dark tutorial is an optional synthetic branch; the shared CCD sequence has
no injected dark current. Its normal reduction uses bias and flat corrections.
