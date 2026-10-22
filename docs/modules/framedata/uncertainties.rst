Uncertainty across a reduction
===============================

FrameData stores standard deviations in the same physical unit as its image.
Variance and inverse variance in supported CCDData inputs are converted on
import. ``uncertainty=None`` means no array is stored; a zero array explicitly
stores zeros. These are different storage states even when an operation treats
missing errors as zero.

.. list-table:: Where uncertainties come from
    :header-rows: 1
    :widths: 25 40 35

    * - Operation
      - Source of output uncertainty
      - Limitation
    * - FrameData creation or FITS read
      - Supplied standard-deviation array
      - Does not infer detector noise from nominal values.
    * - ``imarith`` and CCD calibration
      - First-order propagation of independent input errors
      - Missing errors contribute zero; shared calibration correlations are
        not tracked, even when using the same operand twice.
    * - Mean or median ``imcombine``
      - Population scatter of accepted samples divided by square root of
        their count
      - Does not propagate input errors; one accepted sample has zero scatter.
    * - Sum ``imcombine``
      - Quadrature input errors when all inputs supply them; otherwise
        scatter-based estimate
      - Normalization changes both sum and uncertainty; see the combination guide.
    * - Image registration
      - Resampled input uncertainty when present
      - Interpolation-induced covariance is not propagated.
    * - Aperture photometry
      - Supplied background noise and source Poisson model
      - Does not automatically use FrameData uncertainty or include the
        uncertainty of the fitted local-sky level.
    * - QFloat arithmetic
      - Independent-input derivative propagation
      - Statistical reductions have separate scatter-based contracts.
    * - Polarimetry
      - Beam-flux errors in normalized differences and fit estimates
      - Inspect fit diagnostics and instrument calibration separately.

For example, subtracting a 100 +/- 2 ADU bias from a 1100 +/- 5 ADU science
pixel yields 1000 +/- sqrt(29) ADU. Dividing by a dimensionless flat of
1 +/- 0.01 gives a variance of 29 + 100. This local example checks that chain::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.processing import subtract_bias, flat_correct

    >>> science = FrameData([[1100.]], uncertainty=5, unit='adu')
    >>> bias = FrameData([[100.]], uncertainty=2, unit='adu')
    >>> flat = FrameData([[1.]], uncertainty=0.01)

    >>> # Bias subtraction and flat division both contribute to the propagated error.
    >>> calibrated = flat_correct(subtract_bias(science, bias), flat)

    >>> print('calibrated.data:', np.round(calibrated.data, 6))
    calibrated.data: [[1000.]]
    >>> print('calibrated.uncertainty:', np.round(calibrated.uncertainty, 6))
    calibrated.uncertainty: [[11.357817]]

Reusing that master bias or flat across exposures introduces shared errors that
the independent-input model cannot track. More exposures do not necessarily
reduce calibration systematics by the square root of their count.

To omit image uncertainty storage and calculations, set
``AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = True`` **before loading or
creating frames**. This is process-wide; it does not disable the photometry
routines' own noise model. See :ref:`optional-frame-arrays` and
:doc:`../utilities/configuration` before changing settings.

Keep the operation-specific contracts nearby:
:doc:`../quantities/index`, :doc:`../image/imarith`,
:doc:`../image/imcombine`, :doc:`../image/register`,
:doc:`../photometry/index`, and :doc:`../polarimetry/index`.
