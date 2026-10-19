.. include:: ../references.txt


Image Arithmetic and Combining
==============================

Image arithmetic (|imarith|)
----------------------------

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

Set ``AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY`` to skip uncertainty extraction
and propagation; the result then has ``uncertainty=None``. Unit handling and
nominal calculations are unchanged. See :ref:`optional-frame-arrays` for the
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

Image combining (|imcombine|)
-----------------------------

|imcombine| combines a nonempty list or tuple of images using ``'median'``
(the default), ``'mean'``, or ``'sum'``. Inputs can be images accepted by
``read_framedata``, such as FrameData, NumPy arrays, FITS paths, image HDUs,
CCDData, and two-dimensional QFloat objects. All images must have the same
shape and equal units; equivalent units are not converted automatically.
Input arrays are left unchanged. Images must already be aligned; WCS is not
propagated to the combined frame::

    >>> images = [FrameData(np.full((2, 2), value), unit='adu')
    ...           for value in (1, 3, 5)]
    >>> combined = imcombine(images, method='mean', dtype='float32')
    >>> np.testing.assert_allclose(combined.data, 3)
    >>> assert combined.unit == u.adu
    >>> combined.dtype.name
    'float32'
    >>> combined.uncertainty.dtype.name
    'float32'

Masks and rejection
^^^^^^^^^^^^^^^^^^^

The MASKED flag and input NaNs exclude individual samples at each output pixel.
Other flag bits alone do not exclude samples. Each chunk uses a shared Boolean
validity mask for nominal values and uncertainties: rejection updates the mask
without overwriting either array. Output pixels with no accepted samples are
masked when flag storage is enabled. Their values and uncertainties use the
selected dtype's missing-value fill (NaN when supported, otherwise zero).
Input flag reasons are not merged into the output.

Both rejection methods are disabled by default:

* ``minmax_clip=(low, high)`` rejects values below ``low`` or above ``high``.
  Use ``None`` for an open bound.
* ``sigma_clip=k`` rejects values outside ``center +/- k * deviation`` along
  the image stack at each pixel. A pair ``(low_sigma, high_sigma)`` sets
  asymmetric limits; either element can be ``None`` to disable that side.

Limits themselves are accepted. Clipping also rejects nonfinite values.
Sigma clipping makes one pass using the original unmasked samples, rather than
iterating until convergence. When both methods are enabled, their rejections
are combined; minmax rejection does not change the sigma statistics::

    >>> stack = [FrameData(np.full((2, 2), value)) for value in (1, 2, 3, 100)]
    >>> clipped = imcombine(stack, method='mean', minmax_clip=(None, 10))
    >>> np.testing.assert_allclose(clipped.data, 2)
    >>> robust = imcombine(stack, method='median', sigma_clip=3,
    ...                    sigma_cen_func='median', sigma_dev_func='mad_std')
    >>> np.testing.assert_allclose(robust.data, 2)

``sigma_cen_func`` accepts ``'median'`` or ``'mean'``. ``sigma_dev_func`` accepts
``'std'`` (population standard deviation) or ``'mad_std'`` (median absolute
deviation multiplied by approximately 1.4826). The function defaults are
``'median'`` and ``'std'``; median with ``'mad_std'`` is a useful choice for
outlier-resistant clipping. Custom callables must accept ``axis`` and ignore
excluded samples. They receive a protected copy representing excluded values
as NaNs for floating-point data or as a mask for integer data. Custom
statistics can allocate more workspace than the built-in blocked reductions.

Combination and uncertainties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`N` be the total number of images, :math:`n` the number of accepted
samples at an output pixel, and :math:`x_i` the accepted values. For :math:`n>0`,
the population scatter is

.. math::

    \bar{x} = \frac{1}{n}\sum_i x_i, \qquad
    s = \sqrt{\frac{1}{n}\sum_i (x_i-\bar{x})^2}.

The current estimators are:

.. list-table:: Combination and standard uncertainty
    :header-rows: 1

    * - Method
      - Nominal value
      - Standard uncertainty
    * - ``mean``
      - :math:`\bar{x}`
      - :math:`s/\sqrt{n}`
    * - ``median``
      - Median of accepted values
      - :math:`s/\sqrt{n}`
    * - ``sum`` with all input uncertainties present
      - :math:`\sum_i x_i`
      - :math:`\sqrt{\sum_i \sigma_i^2}`
    * - ``sum`` with any input uncertainty absent
      - :math:`\sum_i x_i`
      - :math:`s\sqrt{n}`

Mean and median estimate uncertainty from sample scatter and do not copy or
propagate individual input uncertainty arrays. The median estimator uses the
same uncertainty formula as the mean, without a median-specific correction.
A single accepted sample therefore has zero scatter-based uncertainty.
For propagated sums, rejected samples are excluded from both sums; NaNs in
accepted input uncertainty arrays contribute zero variance.

For ``method='sum'``, ``sum_normalize=True`` (default) multiplies both the raw
sum and its uncertainty by :math:`N/n`. This compensates for different accepted
sample counts across the image. Pass ``sum_normalize=False`` to obtain the raw
sum and its corresponding uncertainty::

    >>> pair = [FrameData([[10.0]], uncertainty=2),
    ...         FrameData([[100.0]], uncertainty=3)]
    >>> normalized = imcombine(pair, method='sum', minmax_clip=(None, 20))
    >>> raw = imcombine(pair, method='sum', minmax_clip=(None, 20),
    ...                 sum_normalize=False)
    >>> np.testing.assert_allclose(normalized.data, 20)
    >>> np.testing.assert_allclose(normalized.uncertainty, 4)
    >>> np.testing.assert_allclose(raw.data, 10)
    >>> np.testing.assert_allclose(raw.uncertainty, 2)

Both uncertainty-disable settings described above also apply to combination.
They avoid uncertainty buffers, result arrays, and error-only scatter
calculations. Clipping still computes any statistics needed for rejection.

Dtype and memory use
^^^^^^^^^^^^^^^^^^^^

``dtype`` selects the storage type for internal input copies, image-stack
buffers and output data and uncertainty arrays. It accepts supported NumPy
integer and floating-point types, dtype objects and dtype strings, and defaults
to ``numpy.float64``. Input values are converted to this storage type when loaded.

Temporary accumulators and statistics may use wider types. Means divide the
promoted sum before conversion to the output type; medians promote the selected
middle values before averaging. Clipping, variance and uncertainty propagation
also use promoted arithmetic. This prevents a stack of 150 images at 1000 ADU
from overflowing a ``uint16`` mean accumulator. The entire stack is never
converted to a wider temporary cube: scatter and quadrature use bounded blocks.
Intermediate precision follows NumPy's reduction and dtype-promotion rules.

Integer data and uncertainty outputs discard fractional parts toward zero only
when the final reduction is stored. Fractional sum-normalization factors are
applied before that conversion. Final conversion follows NumPy's normal casting
rules, including overflow when a result cannot fit the selected type. Select a
storage dtype with sufficient range. NumPy's floating calculations can also lose
precision for large 64-bit integer values. Floating outputs retain the precision
and range of their selected storage type.

The item size still controls memory for the stack. Promoted work blocks need
additional scratch space, which is included in their block-size estimate.
Sample counts and Boolean validity masks have their own bookkeeping types.

``memory_limit`` is a positive, finite budget in bytes used to size image
chunks and statistics work blocks. Median selection and scatter statistics
process small blocks of output pixels with contiguous samples, bounding their
additional workspace. Internal input copies, output arrays, Python objects,
and some temporaries are outside the budget: it does not cap total process
memory. At least one output pixel's stack must fit in working storage, even
when the requested budget is smaller.

The combiner loads its internal input copies before reducing chunks. For
hundreds of images, consider a smaller dtype and explicit disk caching::

    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     cached = imcombine(images, method='median', dtype='float32',
    ...                        memory_limit=16 * 1024**2,
    ...                        use_disk_cache=True, tmp_dir=directory)
    ...     np.testing.assert_allclose(cached.data, 3)

``use_disk_cache=True`` stores internal image copies in memory-mapped temporary
files, trading disk I/O for reduced resident input storage. Caller-owned frames
remain allocated; passing FITS paths avoids retaining an additional list of
caller-owned image arrays. Present arrays are loaded before being mapped, so
this is not streaming FITS input. Mapped working arrays are released after
combination, including failures. Use a dedicated ``tmp_dir`` because the cache
manager can remove that directory during cleanup. If omitted, a temporary
directory is created automatically when disk caching is enabled.

Headers and reusable configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``merge_header``, ``'no_merge'`` (default) starts with an empty header.
``'first'`` copies the first input header. ``'only_equal'`` keeps keys present
with equal values in every input header. ``'selected_keys'`` keeps keys named
in ``merge_header_keys``, taking each key's first occurrence. The output always
records ``HIERARCH astropop imcombine nimages`` and
``HIERARCH astropop imcombine method``::

    >>> preserved = imcombine([image, image], merge_header='selected_keys',
    ...                          merge_header_keys=['OBJECT'])
    >>> assert preserved.header['OBJECT'] == 'Example'
    >>> assert preserved.header['astropop imcombine nimages'] == 2

Use |ImCombiner| to retain settings for several combinations. Its constructor
uses ``max_memory`` for the budget. Unlike the convenience function,
``set_sigma_clip`` defaults to ``dev_func='mad_std'``::

    >>> combiner = ImCombiner(max_memory=16 * 1024**2, dtype='float32')
    >>> combiner.set_sigma_clip(3, center_func='median', dev_func='mad_std')
    >>> combiner.set_minmax_clip(min_value=0)
    >>> combiner.set_merge_header('first')
    >>> result = combiner.combine(images, method='median')
    >>> np.testing.assert_allclose(result.data, 3)

API
---

.. automodapi:: astropop.image.imarith
    :no-inheritance-diagram:

.. automodapi:: astropop.image.imcombine
    :no-inheritance-diagram:
