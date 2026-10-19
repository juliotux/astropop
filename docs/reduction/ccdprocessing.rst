.. include:: ../references.txt

CCD image processing
====================

:mod:`astropop.image.processing` provides individual calibration steps. It does
not discover calibration files, build masters, or choose their order. Use
:func:`~astropop.image.imcombine.imcombine` to construct masters and apply the
steps appropriate to your detector. Inputs must have matching geometry and
compatible units; these functions do not register calibration frames.

Bias, dark, and flat correction
-------------------------------

``subtract_bias`` subtracts an already combined bias. ``subtract_dark`` scales
an already bias-corrected dark by ``image_exposure / dark_exposure`` before
subtraction. Supply a nonzero dark exposure, with both exposures expressed in
the same units. Exposure values are supplied explicitly rather than read from
the headers::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.processing import subtract_bias, subtract_dark, flat_correct
    >>> science = FrameData(np.full((4, 4), 1200.), unit='adu', uncertainty=5.)
    >>> bias = FrameData(np.full((4, 4), 100.), unit='adu', uncertainty=1.)
    >>> dark = FrameData(np.full((4, 4), 10.), unit='adu', uncertainty=0.5)
    >>> corrected = subtract_bias(science, bias)
    >>> corrected = subtract_dark(corrected, dark, dark_exposure=10, image_exposure=20)
    >>> np.testing.assert_allclose(corrected.data, 1080)

``flat_correct`` divides by a master flat. With ``min_value`` it first replaces
smaller flat values by that lower bound; this is a clamp, not a mask. With
``norm_value`` it divides the flat by that explicitly supplied normalization
value. Neither option estimates a normalization automatically. The master flat
is copied before these changes::

    >>> flat = FrameData(np.full((4, 4), 2.))
    >>> calibrated = flat_correct(corrected, flat, norm_value=2.)
    >>> np.testing.assert_allclose(calibrated.data, 1080)
    >>> assert str(calibrated.unit) == 'adu'

Choose a dimensionless normalized flat to preserve the science unit. A flat
with a physical unit changes the output unit according to division.

Gain and uncertainty handling
-----------------------------

``gain_correct(image, gain)`` multiplies the image by the supplied gain. A
quantity such as ``2 * u.electron / u.adu`` converts ADU to electrons; a bare
number only changes numerical values. An uncertain gain can be supplied as a
``QFloat``. These arithmetic corrections propagate independent input errors
through :func:`~astropop.image.imarith.imarith`; they do not add a detector noise
model when uncertainties are missing.

Bias, dark, flat, and gain corrections preserve the first image's header and
add correction provenance, but currently use ``merge_flags='no_merge'``.
Consequently their arithmetic results have zero flags when storage is enabled;
input masks do not exclude pixels from the calculations. If your workflow must
retain input quality flags, explicitly retain/merge them, or use ``imarith``
with the required flag strategy. Configure optional-array settings before
loading frames; see :ref:`optional-frame-arrays`.

Cropping and cosmic rays
------------------------

``trim_image`` uses ``x_slice`` for columns and ``y_slice`` for rows, following
Python's zero-based, stop-exclusive slicing. It crops data, uncertainties, and
flags together and slices an existing WCS::

    >>> from astropop.image.processing import trim_image
    >>> trimmed = trim_image(calibrated, x_slice=slice(1, 3), y_slice=slice(0, 3))
    >>> trimmed.shape
    (3, 2)

``cosmics_lacosmic`` calls ``astroscrappy.detect_cosmics`` and replaces detected
pixels with the cleaned data. It adds ``COSMIC_RAY | INTERPOLATED`` bits at
those pixels when flag storage is enabled; it does not add ``MASKED`` or
recalculate uncertainties. Keyword arguments are forwarded to Astroscrappy,
so supply its detector/noise parameters explicitly. Existing FrameData masks
are not automatically forwarded as an Astroscrappy input mask.

Every routine defaults to returning a separate frame. ``inplace=True`` updates
the supplied FrameData; conversion of other supported input types still
creates a FrameData. Record calibration choices outside the library when
reproducibility requires more than the stored correction keywords.

CCD processing API
------------------

.. automodapi:: astropop.image.processing
    :no-inheritance-diagram:
