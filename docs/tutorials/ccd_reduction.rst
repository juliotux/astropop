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

Extract the archive once. It includes ``data/``, nine numbered ``notebooks/``,
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
:doc:`../concepts/configuration` for optional storage.

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
clipping. Review :doc:`registration_diagnostics` for center/edge
residuals, and :doc:`../how_to/image_combination` for estimator and rejection
choices. Registration, outlier rejection and distortion correction are distinct.

Save and inspect the products
------------------------------

Follow :doc:`photometry` for the stack, or :doc:`light_curve` for measurements on
individual exposures. :doc:`astrometric_calibration` attaches a fitted WCS using
local standards; :doc:`photometric_calibration` then converts instrumental
fluxes to a stated magnitude convention. :doc:`polarimetry` uses a separate
paired-beam input rather than treating ordinary images as polarimetric data.

Inspect figures and printed output at each stage. The examples overwrite named
derived products so notebooks can be rerun; raw files remain inputs. Source
truth is used only for recovery diagnostics. A small residual in the simulation
does not establish accuracy for different detectors or observations.
