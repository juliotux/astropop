.. include:: ../references.txt

Image arithmetic
================

Use |imarith| to calculate with image values, physical units, and standard
uncertainties. It returns a |FrameData| and supports ``+``, ``-``, ``*``, ``/``,
``//``, ``%``, and ``**``. Operands may be |FrameData|, |QFloat|, |Quantity|,
|CCDData|, numerical arrays, or numbers. A unit object or unit string represents
one of that unit. NumPy broadcasting applies to the numerical operands::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData, PixelMaskFlags
    >>> from astropop.image import imarith, ImArith, imcombine, ImCombiner
    >>> image = FrameData(np.full((2, 2), 10.0), unit='adu', uncertainty=2.0,
    ...                   meta={'OBJECT': 'Example', 'EXPTIME': 30.0})
    >>> scaled = imarith(image, 2, '*')
    >>> np.testing.assert_allclose(scaled.data, 20)
    >>> np.testing.assert_allclose(scaled.uncertainty, 4)
    >>> assert scaled.unit == image.unit
    >>> assert scaled is not image

Nominal calculations use floating-point values. This function has no ``dtype``
option; use |imcombine| for combination with a selected working dtype. Neither
operation registers images: align images before doing pixel-by-pixel reduction.

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
    >>> added = imarith(lengths, 50 * u.cm, '+')
    >>> np.testing.assert_allclose(added.data, 1.5)
    >>> np.testing.assert_allclose(added.uncertainty, 0.1)
    >>> assert added.unit == u.m

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
    >>> masked_result = imarith(image, 2, '*')
    >>> assert masked_result.mask[0, 0]
    >>> assert masked_result.data[0, 0] == 20

For header merging, ``'only_equal'`` (default) keeps keys present with equal
values in both headers. ``'first'`` copies the first operand's header.
``'selected_keys'`` keeps keys named in ``selected_keys``, taking each key's
first occurrence. ``'no_merge'`` returns an empty header::

    >>> selected = imarith(image, 2, '*', merge_headers='selected_keys',
    ...                   selected_keys=['OBJECT'])
    >>> assert selected.header['OBJECT'] == 'Example'
    >>> assert 'EXPTIME' not in selected.header

By default, operands remain unchanged. ``inplace=True`` updates and returns the
first operand when it is FrameData; otherwise a new FrameData is returned.
The updated frame receives the result's data, unit, uncertainty, flags, and
merged header. Skipping propagation therefore clears an in-place result's old
uncertainty. Arithmetic does not propagate WCS to a new result.

Use ``ImArith`` when reusing merge settings across several calls::

    >>> arithmetic = ImArith(merge_headers='first')
    >>> result = arithmetic.run(image, 2, '*')
    >>> assert result.header['OBJECT'] == 'Example'


API reference: :doc:`../reference/image`.
