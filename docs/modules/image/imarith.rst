.. include:: ../../references.txt

Image arithmetic
================

The :mod:`astropop.image.imarith` module exposes ``imarith`` for a single
calculation and ``ImArith`` for repeated calculations with shared merge settings.
Both normalize operands, apply the selected operation, propagate enabled
uncertainties and construct a FrameData result. This chapter describes those
interfaces; :doc:`../../tutorials/ccd_reduction/science_reduction` combines image operations in
a complete reduction.

The ``imarith`` function
------------------------

Use |imarith| to calculate with image values, physical units, and standard
uncertainties. It returns a |FrameData| and supports ``+``, ``-``, ``*``, ``/``,
``//``, ``%``, and ``**``. Operands may be |FrameData|, |QFloat|, |Quantity|,
|CCDData|, numerical arrays, or numbers. A unit object or unit string represents
one of that unit. NumPy broadcasting applies to the numerical operands::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData, PixelMaskFlags
    >>> from astropop.image import imarith, ImArith

    >>> image = FrameData(np.full((2, 2), 10.0), unit='adu', uncertainty=2.0,
    ...                   meta={'OBJECT': 'Example', 'EXPTIME': 30.0})
    >>> scaled = imarith(image, 2, '*')

    >>> print('scaled.data:', np.round(scaled.data, 6))
    scaled.data: [[20. 20.]
     [20. 20.]]
    >>> print('scaled.uncertainty:', np.round(scaled.uncertainty, 6))
    scaled.uncertainty: [[4. 4.]
     [4. 4.]]
    >>> print('scaled.unit:', scaled.unit)
    scaled.unit: adu
    >>> print('Input pixels:', image.data)
    Input pixels: [[10. 10.]
     [10. 10.]]

Nominal calculations use floating-point values; ``imarith`` has no ``dtype``
argument. NumPy broadcasting determines the result shape; unlike direct
FrameData construction, this arithmetic path also supports scalar results.
The operation acts on corresponding numerical elements; it does not estimate
an alignment or use WCS to match the operands.

Units and uncertainty propagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Addition and subtraction convert the second operand to the first operand's
unit. Incompatible units raise ``UnitsError``. Multiplication and division
combine units; powers require a dimensionless scalar exponent. Floor division
and remainder operate on the operands' raw numerical values without converting
the second unit first. Floor division returns the quotient unit; remainder
keeps the first operand's unit::

    >>> from astropy import units as u

    >>> lengths = FrameData(np.ones((2, 2)), unit='m', uncertainty=0.1)

    >>> # Addition converts centimeters to the first operand's meter unit.
    >>> added = imarith(lengths, 50 * u.cm, '+')

    >>> print('added.data:', np.round(added.data, 6))
    added.data: [[1.5 1.5]
     [1.5 1.5]]
    >>> print('added.uncertainty:', np.round(added.uncertainty, 6))
    added.uncertainty: [[0.1 0.1]
     [0.1 0.1]]
    >>> print('added.unit:', added.unit)
    added.unit: m

Uncertainties are propagated with the derivative rules used by |QFloat|,
assuming independent operands. For a result :math:`f(a,b)`, the variance is

.. math::

    \sigma_f^2 = \left(\frac{\partial f}{\partial a}\sigma_a\right)^2
               + \left(\frac{\partial f}{\partial b}\sigma_b\right)^2.

An operand with no uncertainty contributes zero to this calculation. With
propagation enabled, even two exact operands produce a stored zero uncertainty
array. Correlations between operands are not tracked, including when the same
frame is passed twice. CCDData variance and inverse-variance uncertainties are
converted to standard deviations before propagation.

.. plot::
    :include-source: false
    :alt: Image-row values before and after bias and flat correction, with the propagated standard uncertainty for each stage below.
    :caption: Subtracting an uncertain bias increases the variance even as it lowers the signal. Dividing by an uncertain flat adds a signal-dependent error, most visible at the star. These noiseless nominal profiles use supplied independent errors of 4 ADU for the raw pixels, 2 ADU for the bias and 0.02 for the dimensionless flat.

    import matplotlib.pyplot as plt
    import numpy as np

    from astropop.framedata import FrameData
    from astropop.image import imarith

    x = np.arange(80)
    signal = 50 + 300 * np.exp(-0.5 * ((x - 42) / 6)**2)
    response = 0.8 + 0.4 * x / (len(x) - 1)
    raw = FrameData((100 + response * signal)[None, :], unit='adu', uncertainty=4.)
    bias = FrameData(np.full(raw.shape, 100.), unit='adu', uncertainty=2.)
    flat = FrameData(response[None, :], uncertainty=0.02)
    subtracted = imarith(raw, bias, '-')

    # Remove the additive offset before dividing by the dimensionless response.
    corrected = imarith(subtracted, flat, '/')
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5.3),
                             layout='constrained')
    for frame, name, color in [(raw, 'Raw', '0.5'),
                                (subtracted, 'Bias subtracted', 'C0'),
                                (corrected, 'Flat corrected', 'C1')]:
        axes[0].plot(x, frame.data[0], color=color, label=name)
        axes[1].plot(x, frame.uncertainty[0], color=color)

    axes[0].plot(x, signal, 'k--', linewidth=1, label='Injected signal')
    axes[0].set(ylabel='Nominal value (ADU)', title='Arithmetic on one image row')
    axes[0].legend(fontsize='small')
    axes[1].set(xlabel='Column x (pixel)', ylabel='Standard uncertainty (ADU)',
                title='Supplied errors propagated as independent inputs')
    for ax in axes:
        ax.grid(alpha=0.2)

Compare nominal values and uncertainties separately: removing an instrumental
offset does not remove its calibration uncertainty. This diagram uses known
bias and flat profiles to isolate the arithmetic; estimating those masters
from exposures is a separate task.

Set ``AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = True`` **before loading or
creating frames** to omit uncertainty storage and skip extraction and
propagation; the result then has ``uncertainty=None``. Setting the option after
loading hides existing uncertainty arrays without freeing their memory. Unit
handling and nominal calculations are unchanged. See :ref:`optional-frame-arrays` for the
storage switches, their effect on existing frames, and a temporary-setting
example.

Flags, headers, and in-place results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only FrameData operands contribute flags and headers. To retain a CCDData
mask or metadata, convert it with ``read_framedata`` before arithmetic.
``merge_flags='or'`` (the default) combines all flag bits by bitwise OR;
``'and'`` keeps bits present in both operands; ``'no_merge'`` produces zero
flags. Missing flags count as zero. Masks mark the output but do not exclude
pixels from arithmetic. Disabling flag storage omits output flags and masks::

    >>> image.mask_pixels((0, 0))

    >>> # The mask marks quality; it does not stop arithmetic on the stored pixel.
    >>> masked_result = imarith(image, 2, '*')

    >>> print('masked_result.mask[0, 0]:', masked_result.mask[0, 0])
    masked_result.mask[0, 0]: True
    >>> print('masked_result.data[0, 0]:', masked_result.data[0, 0])
    masked_result.data[0, 0]: 20.0

For header merging, ``'only_equal'`` (default) keeps keys present with equal
values in both headers. ``'first'`` copies the first operand's header.
``'selected_keys'`` keeps keys named in ``selected_keys``, taking each key's
first occurrence. ``'no_merge'`` returns an empty header::

    >>> selected = imarith(image, 2, '*', merge_headers='selected_keys',
    ...                   selected_keys=['OBJECT'])

    >>> print("selected.header['OBJECT']:", selected.header['OBJECT'])
    selected.header['OBJECT']: Example
    >>> print('Exposure retained:', 'EXPTIME' in selected.header)
    Exposure retained: False

By default, operands remain unchanged. ``inplace=True`` updates and returns the
first operand when it is FrameData; otherwise a new FrameData is returned.
The updated frame receives the result's data, unit, uncertainty, flags, and
merged header. Skipping propagation therefore clears an in-place result's old
uncertainty. Arithmetic does not propagate WCS to a new result.

The ``ImArith`` class
---------------------

``ImArith`` stores ``merge_flags``, ``merge_headers`` and ``selected_keys``.
Its ``run(operand1, operand2, operation, inplace=False)`` method performs the
same calculation as ``imarith``. The class does not retain a source image;
pass operands to each call. Optional-array configuration is read for each run,
so the class uses the current FrameData uncertainty and flag policy.

Create an instance with a fixed header-merge strategy, then call ``run``::

    >>> arithmetic = ImArith(merge_headers='first')
    >>> result = arithmetic.run(image, 2, '*')

    >>> print("result.header['OBJECT']:", result.header['OBJECT'])
    result.header['OBJECT']: Example




Input and result diagnostics
----------------------------

.. list-table:: Symptoms, checks and recovery
    :header-rows: 1
    :widths: 24 40 36

    * - Symptom
      - Minimal diagnostic
      - Recovery and contract
    * - Addition/subtraction raises a units error
      - Inspect both operand units, including bare numbers being dimensionless.
      - Supply a compatible Quantity or calibrated units; do not erase units
        to bypass the check. See :doc:`imarith`.

Reference/API
-------------

.. automodapi:: astropop.image.imarith
    :no-inheritance-diagram:


.. currentmodule:: None
