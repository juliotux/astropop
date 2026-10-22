.. include:: ../../references.txt

CCD image processing
====================

:mod:`astropop.image.processing` supplies ``subtract_bias``, ``subtract_dark``,
``flat_correct`` and ``gain_correct`` for calibration arithmetic, ``trim_image``
for cropping, and ``cosmics_lacosmic`` for cosmic-ray cleaning. Each function
accepts image inputs through FrameData conversion and returns a FrameData.
The arithmetic functions accept ``inplace=True`` to update the science frame;
their examples below use the default independent result.

This module applies supplied calibrations; it does not select observations or
estimate masters. For that complete process, see
:doc:`../../tutorials/ccd_reduction/science_reduction`. The sections below explain the numerical
steps, options and returned image information.

The additive bias/dark and multiplicative flat model is described in Astropy's
`CCD calibration overview
<https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-05-Calibration-overview.html>`_.
The functions below implement individual arithmetic steps in that model;
the application must supply compatible masters, exposure information and a
detector-appropriate uncertainty model.

Bias, dark, and flat correction
-------------------------------

The ``subtract_bias`` function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``subtract_bias(image, master_bias)`` computes :math:`C=I-B`, where :math:`I`
is the science image and :math:`B` the supplied master bias. Pixel geometry and
units must be compatible. Enabled standard uncertainties propagate using the
independent-input rule :math:`\sigma_C^2=\sigma_I^2+\sigma_B^2`.

Subtract a uniform 100 ADU bias from a 1200 ADU image::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.processing import subtract_bias, subtract_dark, flat_correct

    >>> science = FrameData(np.full((4, 4), 1200.), unit='adu', uncertainty=5.)
    >>> bias = FrameData(np.full((4, 4), 100.), unit='adu', uncertainty=1.)
    >>> dark = FrameData(np.full((4, 4), 10.), unit='adu', uncertainty=0.5)
    >>> corrected = subtract_bias(science, bias)

    >>> print(f'Bias-subtracted pixel: {corrected.data[0, 0]:.0f} {corrected.unit}')
    Bias-subtracted pixel: 1100 adu

The ``subtract_dark`` function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``subtract_dark`` scales a supplied bias-corrected master dark by
``image_exposure / dark_exposure`` and subtracts it:

.. math::

    C = I - \frac{t_I}{t_D}D.

Exposure values are explicit arguments, not inferred from headers. Supply a
nonzero dark exposure and use the same time unit for both arguments. A ten-ADU
dark measured in ten seconds removes twenty ADU from a twenty-second image::

    >>> # Scale the 10-second dark to the science exposure's 20 seconds.
    >>> corrected = subtract_dark(corrected, dark, dark_exposure=10, image_exposure=20)

    >>> print('corrected.data:', np.round(corrected.data, 6))
    corrected.data: [[1080. 1080. 1080. 1080.]
     [1080. 1080. 1080. 1080.]
     [1080. 1080. 1080. 1080.]
     [1080. 1080. 1080. 1080.]]

Master construction and exposure-group selection are demonstrated in
:doc:`../../tutorials/ccd_reduction/dark_correction`. The scaling assumes a dark signal
proportional to exposure time; the function does not fit a temperature or
nonlinearity model.

The ``flat_correct`` function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``flat_correct`` divides by a master flat. With ``min_value`` it first replaces
smaller flat values by that lower bound; this is a clamp, not a mask. With
``norm_value`` it divides the flat by that explicitly supplied normalization
value. Neither option estimates a normalization automatically. With a supplied
normalization :math:`n`, the calculation is :math:`C=I/(F/n)` after any lower
bound has been applied to :math:`F`. The master flat is copied before these
changes::

    >>> flat = FrameData(np.full((4, 4), 2.))
    >>> calibrated = flat_correct(corrected, flat, norm_value=2.)

    >>> print('calibrated.data:', np.round(calibrated.data, 6))
    calibrated.data: [[1080. 1080. 1080. 1080.]
     [1080. 1080. 1080. 1080.]
     [1080. 1080. 1080. 1080.]
     [1080. 1080. 1080. 1080.]]
    >>> print('str(calibrated.unit):', str(calibrated.unit))
    str(calibrated.unit): adu

Choose a dimensionless normalized flat to preserve the science unit. A flat
with a physical unit changes the output unit according to division.

.. plot::
    :include-source: false
    :alt: Raw stellar field, the same field after bias and dark subtraction, the normalized flat with a sensitivity gradient and dust shadow, and the final corrected field.
    :caption: A synthetic field corrected with subtract_bias, subtract_dark and flat_correct. The ideal bias and bias-corrected dark are known here; the dark is scaled from 10 to 20 seconds. The flat is dimensionless and normalized by its median. The three ADU panels share display limits; bright cores exceed the displayed range but are not saturated in the data.

    import matplotlib.pyplot as plt
    import numpy as np
    from astropop.framedata import FrameData
    from astropop.image.processing import subtract_bias, subtract_dark, flat_correct

    y, x = np.indices((96, 96))
    signal = 100 + 400 * np.exp(-((x - 40)**2 + (y - 53)**2) / (2 * 3**2))
    signal += 250 * np.exp(-((x - 72)**2 + (y - 22)**2) / (2 * 3**2))
    bias = FrameData(100 + 0.4 * x + 3 * np.sin(y / 7), unit='adu')
    dark = FrameData(5 + 0.03 * y, unit='adu')
    response = (0.7 + 0.6 * x / 95) * (1 - 0.18 * np.exp(
        -((x - 28)**2 + (y - 28)**2) / (2 * 10**2)))
    response /= np.median(response)
    flat = FrameData(response)
    rng = np.random.default_rng(42)

    # Simulate a science exposure twice as long as the dark master.
    raw = FrameData(bias.data + 2 * dark.data + response * signal
                    + rng.normal(0, 2, x.shape), unit='adu')
    subtracted = subtract_dark(subtract_bias(raw, bias), dark,
                               dark_exposure=10, image_exposure=20)
    corrected = flat_correct(subtracted, flat)
    panels = [(raw.data, 'Raw: offset + dark + response', 'ADU', 80, 300),
              (subtracted.data, 'Bias and scaled dark subtracted', 'ADU', 80, 300),
              (flat.data, 'Normalized flat response', 'Dimensionless', 0.65, 1.35),
              (corrected.data, 'After flat correction', 'ADU', 80, 300)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout='constrained')
    for ax, (pixels, title, unit, low, high) in zip(axes.flat, panels):
        image = ax.imshow(pixels, origin='lower', cmap='viridis', vmin=low, vmax=high)
        fig.colorbar(image, ax=ax, label=unit, shrink=0.8,
                     extend='max' if unit == 'ADU' else 'neither')
        ax.set(title=title, xlabel='x (pixel)', ylabel='y (pixel)')

The residual gradient and dust shadow after subtraction are multiplicative
response effects, so they remain until division by the flat. The final sky is
approximately uniform, with the supplied random noise still present. These
ideal calibration inputs demonstrate the operations, not the accuracy of
masters estimated from a finite observing run. Follow
:doc:`../../tutorials/ccd_reduction/master_bias` and :doc:`../../tutorials/ccd_reduction/master_flat` to build
masters from exposures.

Gain and uncertainty handling
-----------------------------

``gain_correct(image, gain)`` multiplies the image by the supplied gain. A
quantity such as ``2 * u.electron / u.adu`` converts ADU to electrons; a bare
number only changes numerical values. An uncertain gain can be supplied as a
``QFloat``. These arithmetic corrections propagate independent input errors
through :func:`~astropop.image.imarith.imarith`; they do not add a detector noise
model when uncertainties are missing.

For a bias-subtracted single exposure in ADU, a simple detector model is
``variance_adu = max(signal_adu, 0) / gain_e_per_adu + (read_noise_e / gain_e_per_adu)**2``.
This assumes independent Poisson photoelectrons and Gaussian read noise, treats
gain as exact, and excludes master-frame uncertainty. Supply that uncertainty
explicitly before converting the image to electrons::

    >>> from astropy import units as u
    >>> from astropop.image.processing import gain_correct

    >>> gain_e_per_adu, read_noise_e = 2., 4.
    >>> signal_adu = np.array([[1000.]])

    >>> # Express both Poisson noise and read noise in ADU before adding variances.
    >>> sigma_adu = np.sqrt(np.maximum(signal_adu, 0) / gain_e_per_adu
    ...                     + (read_noise_e / gain_e_per_adu)**2)
    >>> measured_signal = FrameData(signal_adu, unit='adu', uncertainty=sigma_adu)
    >>> electrons = gain_correct(measured_signal, gain_e_per_adu * u.electron / u.adu)

    >>> print(f'Signal: {electrons.data[0, 0]:.0f} {electrons.unit}')
    Signal: 2000 electron
    >>> print(f'Standard uncertainty: {electrons.uncertainty[0, 0]:.2f} {electrons.unit}')
    Standard uncertainty: 44.90 electron

Clipping negative signal to zero in this variance estimate does not change the
image pixels. Include sky photoelectrons in the Poisson term even if a later
step subtracts the sky. Add the appropriate calibration variances for actual
observations; gain conversion itself only multiplies existing values/errors.

Bias, dark, flat, and gain corrections preserve the first image's header and
add correction provenance, but currently use ``merge_flags='no_merge'``.
Consequently their arithmetic results have zero flags when storage is enabled;
input masks do not exclude pixels from the calculations. If your workflow must
retain input quality flags, explicitly retain/merge them, or use ``imarith``
with the required flag strategy. Configure optional-array settings before
loading frames; see :ref:`optional-frame-arrays`.

Low flat response and quality maps
----------------------------------

A zero response cannot be inverted scientifically. Clamping a normalized flat
can keep intermediate divisions finite, but the affected pixels still need a
quality policy. This illustrative threshold of 0.1 rejects both the low-response
and zero-response pixels and explicitly restores the merged flags::

    >>> from astropop.framedata import PixelMaskFlags as F

    >>> exposure = FrameData([[100., 100., 100.]], unit='adu')
    >>> response = FrameData([[1., 0.02, 0.]])
    >>> response.add_flags(F.MASKED | F.BAD, where=response.data < 0.1)

    >>> # Retain the reason bits explicitly across the calibration helper.
    >>> quality = np.bitwise_or(exposure.flags, response.flags)
    >>> divided = flat_correct(exposure, response, min_value=0.1)

    >>> # A denominator floor avoids extreme values; the saved flags still exclude defects.
    >>> divided.flags = quality

    >>> print('Finite intermediate values:', divided.data.tolist())
    Finite intermediate values: [[100.0, 1000.0, 1000.0]]
    >>> print('Excluded pixels:', divided.mask.tolist())
    Excluded pixels: [[False, True, True]]
    >>> print('Original response:', response.data.tolist())
    Original response: [[1.0, 0.02, 0.0]]

Do not use the finite values at excluded pixels as repaired measurements.
Choose response limits using the actual detector; inspect nonfinite responses
too. See :doc:`../../tutorials/diagnostics/quality_propagation` for defects in both science and masters,
and full reason-bit persistence. ``min_value`` alone does not mark a mask or
update the uncertainty model for clamped flat values.

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

For example, clean a single sharp event in an otherwise uniform frame. This
tests replacement and flag handling without simulating a real stellar field::

    >>> from astropop.image.processing import cosmics_lacosmic

    >>> pixels = np.full((32, 32), 100.)
    >>> pixels[16, 16] = 10000.
    >>> event = FrameData(pixels, unit='adu', uncertainty=4.)
    >>> cleaned = cosmics_lacosmic(event, gain=1., readnoise=4., niter=1)

    >>> print(f'Cleaned pixel: {cleaned.data[16, 16]:.1f} ADU')
    Cleaned pixel: 100.0 ADU
    >>> print('Cosmic-ray flag:', cleaned.mask_flags(F.COSMIC_RAY)[16, 16])
    Cosmic-ray flag: True
    >>> print('Interpolation flag:', cleaned.mask_flags(F.INTERPOLATED)[16, 16])
    Interpolation flag: True
    >>> print('Excluded from later combination:', cleaned.mask[16, 16])
    Excluded from later combination: False

Choose explicitly whether to keep an interpolated pixel or add ``MASKED`` to
exclude it. Its stored uncertainty is still the input value; replacement does
not estimate the interpolation error. The underlying method uses Laplacian edge
detection, described by `van Dokkum (2001), PASP 113, 1420
<https://arxiv.org/abs/astro-ph/0108003>`_. Detection thresholds and PSF sampling
must be checked on observations; this isolated event does not establish the
false-positive rate for stars or spectra.

Every routine defaults to returning a separate frame. ``inplace=True`` updates
the supplied FrameData; conversion of other supported input types still
creates a FrameData. Record calibration choices outside the library when
reproducibility requires more than the stored correction keywords.

Overscan subtraction and detector nonlinearity correction are outside these
convenience routines. Apply the instrument's calibrated correction and propagate
its uncertainties at the appropriate stage. Use the same overscan, trimming,
readout and linearity conventions for masters and science; do not interpret
bias subtraction as a replacement for those detector-specific steps.




Reference/API
-------------

.. automodapi:: astropop.image.processing
    :no-inheritance-diagram:


.. currentmodule:: None
