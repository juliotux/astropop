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

The diagrams below separate the shared calibration products from the two
imaging workflows. Arrows indicate which products a step needs; follow the
linked tutorials for the actual calculations.

.. mermaid::
    :align: center
    :alt: Raw bias exposures produce a master bias. Raw flats and the master bias feed bias subtraction and normalization, which produce the master flat.
    :caption: Build the shared calibration masters. The normalized flat is derived from bias-corrected exposures.

    flowchart TD
        accTitle: Shared calibration masters
        accDescr: Raw bias exposures produce a master bias. Raw flats and the master bias feed bias subtraction and normalization, which produce the master flat.
        biases["Raw bias exposures"] --> bias["Master bias"]
        bias --> correct["Subtract bias and normalize<br/>each flat exposure"]
        flats["Raw flat exposures"] --> correct
        correct -->|combine| flat["Master flat"]

.. mermaid::
    :align: center
    :alt: Master bias, master flat and raw science exposures feed science calibration. Calibrated images are registered and combined, then measured. Local references and source measurements provide astrometric and photometric calibration.
    :caption: Produce and measure a science stack. Reference stars supply the local astrometric and photometric standards.

    flowchart TD
        accTitle: Science stack and photometric calibration
        accDescr: Master bias, master flat and raw science exposures feed science calibration. Calibrated images are registered and combined, then measured. Local references and source measurements provide astrometric and photometric calibration.
        masters["Master bias + master flat"] --> calibrated["Calibrate individual exposures"]
        raw["Raw science exposures"] --> calibrated
        calibrated --> stack["Register and combine"]
        stack --> measured["Detect stars and measure fluxes"]
        measured --> wcs["Fit the astrometric solution"]
        references["Local reference stars"] --> wcs
        wcs --> magnitudes["Calibrate magnitudes"]
        references --> magnitudes

.. mermaid::
    :align: center
    :alt: The same master bias and flat calibrate a separate twelve-exposure sequence. Measurements on individual exposures produce the differential light curve without using the science stack.
    :caption: The light curve uses a separate observing sequence and retains individual exposures rather than measuring the science stack.

    flowchart TD
        accTitle: Differential time-series photometry
        accDescr: The same master bias and flat calibrate a separate twelve-exposure sequence. Measurements on individual exposures produce the differential light curve without using the science stack.
        masters["Same master bias + master flat"] --> calibrated["Calibrate individual exposures"]
        sequence["Separate raw time series<br/>12 exposures"] --> calibrated
        calibrated --> curve["Measure target and comparison stars<br/>and produce a differential light curve"]

Polarimetry uses separate half-wave images or a quarter-wave flux table. These
inputs do not come from the imaging stack; see :doc:`../polarimetry/index`.


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
