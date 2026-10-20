.. include:: ../references.txt

Image combination
=================

Start with three small images to inspect combination values and units::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropop.framedata import FrameData
    >>> from astropop.image import imcombine, ImCombiner

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
    >>> print('combined.data:', np.round(combined.data, 6))
    combined.data: [[3. 3.]
     [3. 3.]]
    >>> print('combined.unit:', combined.unit)
    combined.unit: adu
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
    >>> print('clipped.data:', np.round(clipped.data, 6))
    clipped.data: [[2. 2.]
     [2. 2.]]
    >>> robust = imcombine(stack, method='median', sigma_clip=3,
    ...                    sigma_cen_func='median', sigma_dev_func='mad_std')
    >>> print('robust.data:', np.round(robust.data, 6))
    robust.data: [[2. 2.]
     [2. 2.]]

``sigma_cen_func`` accepts ``'median'`` or ``'mean'``. ``sigma_dev_func`` accepts
``'std'`` (population standard deviation) or ``'mad_std'`` (median absolute
deviation multiplied by approximately 1.4826). The function defaults are
``'median'`` and ``'std'``; median with ``'mad_std'`` is a useful choice for
outlier-resistant clipping. Custom callables must accept ``axis`` and ignore
excluded samples. They receive a protected copy representing excluded values
as NaNs for floating-point data or as a mask for integer data. Custom
statistics can allocate more workspace than the built-in blocked reductions.

Inspect a transient in a noisy stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A fixed seed makes this small bias example reproducible. Nine exposures share
an injected level of 1000 ADU with 3 ADU Gaussian noise. One sample receives a
600 ADU event; the other exposures retain an ordinary value at that pixel::

    >>> rng = np.random.default_rng(42)
    >>> pixels = rng.normal(1000, 3, (9, 48, 48))
    >>> pixels[0, 24, 24] += 600
    >>> bias_frames = [FrameData(image, unit='adu') for image in pixels]
    >>> estimates = {
    ...     'Mean': imcombine(bias_frames, method='mean'),
    ...     'Median': imcombine(bias_frames, method='median'),
    ...     'Clipped mean': imcombine(bias_frames, method='mean', sigma_clip=4,
    ...                               sigma_dev_func='mad_std'),
    ... }
    >>> for name, image in estimates.items():
    ...     print(f'{name}: {image.data[24, 24]:.1f} ADU')
    Mean: 1067.1 ADU
    Median: 1001.6 ADU
    Clipped mean: 1000.8 ADU

.. plot:: _ext/guide_figures.py plot_combination
    :include-source: False
    :alt: Nine samples at a contaminated pixel with three combination estimates, followed by histograms of residuals from the injected bias level.
    :caption: Left: the event biases the ordinary mean at one pixel. Right: residual histograms across the output images; logarithmic counts and compressed horizontal tails keep both the central distribution and the mean's single affected pixel visible.

The median and clipped mean stay near the injected level at the affected pixel.
Their values need not equal exactly 1000 ADU because only a few noisy samples
are available. Clipping can also reject legitimate noise samples, especially
when few exposures give an unstable deviation estimate. Choose thresholds using
representative observations; a narrower histogram alone does not demonstrate
better calibration or justify the supplied uncertainty model. Registration
errors, variable sources and detector defects need separate treatment.

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
    >>> print('normalized.data:', np.round(normalized.data, 6))
    normalized.data: [[20.]]
    >>> print('normalized.uncertainty:', np.round(normalized.uncertainty, 6))
    normalized.uncertainty: [[4.]]
    >>> print('raw.data:', np.round(raw.data, 6))
    raw.data: [[10.]]
    >>> print('raw.uncertainty:', np.round(raw.uncertainty, 6))
    raw.uncertainty: [[2.]]

``AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY`` also applies to combination.
Set it before loading or creating frames to avoid input uncertainty storage,
combination uncertainty buffers, output uncertainty arrays, and error-only
scatter calculations. Clipping still computes statistics needed for rejection.
``FRAMEDATA_DISABLE_FLAGS`` independently disables flag and mask storage; it
does not disable uncertainty calculations::

    >>> from astropop.config import AstropopConfig as conf
    >>> previous_uncertainty = conf.FRAMEDATA_DISABLE_UNCERTAINTY
    >>> previous_flags = conf.FRAMEDATA_DISABLE_FLAGS
    >>> try:
    ...     conf.FRAMEDATA_DISABLE_UNCERTAINTY = True
    ...     conf.FRAMEDATA_DISABLE_FLAGS = True
    ...     inputs = [FrameData([[2.0]], uncertainty=1),
    ...               FrameData([[4.0]], uncertainty=1)]
    ...     combined = imcombine(inputs, method='mean')
    ...     print('combined.data:', np.round(combined.data, 6))
    ...     print('combined.uncertainty:', combined.uncertainty)
    ...     print('combined.flags:', combined.flags)
    ...     print('combined.mask:', combined.mask)
    ...     conf.FRAMEDATA_DISABLE_UNCERTAINTY = False
    ...     conf.FRAMEDATA_DISABLE_FLAGS = False
    ...     enabled = imcombine([FrameData([[2.0]]), FrameData([[4.0]])],
    ...                         method='mean')
    ...     print('enabled.data:', np.round(enabled.data, 6))
    ...     print('enabled.uncertainty:', enabled.uncertainty)
    ...     print('enabled.flags:', enabled.flags)
    ... finally:
    ...     conf.FRAMEDATA_DISABLE_UNCERTAINTY = previous_uncertainty
    ...     conf.FRAMEDATA_DISABLE_FLAGS = previous_flags
    combined.data: [[3.]]
    combined.uncertainty: None
    combined.flags: None
    combined.mask: None
    enabled.data: [[3.]]
    enabled.uncertainty: [[0.70710678]]
    enabled.flags: [[0]]

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
    ...     print('cached.data:', np.round(cached.data, 6))
    cached.data: [[3. 3.]
     [3. 3.]]

``use_disk_cache=True`` stores internal image copies in memory-mapped temporary
files, trading disk I/O for reduced resident input storage. Caller-owned frames
remain allocated; passing FITS paths avoids retaining an additional list of
caller-owned image arrays. Present arrays are loaded before being mapped, so
this is not streaming FITS input. Mapped working arrays are released after
combination, including failures. Use a dedicated ``tmp_dir`` because the cache
manager can remove that directory during cleanup. If omitted, a temporary
directory is created automatically when disk caching is enabled.

.. _imcombine-memory-planning:

Plan memory for many images
"""""""""""""""""""""""""""

Choose storage dtype and optional arrays before loading data. A stack of
``N`` images with ``P`` pixels and item size ``b`` needs ``N*P*b`` bytes per
stored stack array. For 150 images of 1024 by 1024 pixels at float32, nominal
input storage alone is 600 MiB. Same-sized uncertainties add another 600 MiB;
one uint8 flag array per image adds 150 MiB. Caller-owned frames and the
combiner's copies can coexist. These estimates exclude objects, headers,
outputs, validity masks, sample counts, promoted scratch buffers and libraries.
The work-tile budget described above does not cap these allocations.

Pass FITS paths instead of retaining a second list of caller-owned image arrays.
Set optional-array configuration before loading frames::

    from astropop.config import AstropopConfig as conf
    from astropop.image import imcombine
    conf.FRAMEDATA_DISABLE_UNCERTAINTY = True  # before loading any frames
    # Set this only if your reduction does not require FrameData masks/flags.
    conf.FRAMEDATA_DISABLE_FLAGS = True
    combined = imcombine(paths, method='median', dtype='float32',
                         memory_limit=64 * 1024**2, use_disk_cache=True)

This is a recipe with application-supplied ``paths``, not an executed benchmark.
Disk caching uses the loading and cleanup behavior described above. Filesystem
page cache, disk space and I/O speed also matter. Never use a shared working
directory as ``tmp_dir``. Caller-owned FrameData storage remains your
responsibility.

Iterate over science products and write each result before processing the next.
For measurements, record input count, image shape, dtype, optional-array policy,
method/rejection settings, cache mode, memory budget, software versions and
hardware. Benchmark in a fresh process and report both elapsed time and peak
resident memory, distinguishing baseline interpreter/import cost from the
increment during combination. Sampling RSS can miss short peaks and includes
mapped pages; allocator tracing alone may omit native allocations. Keep those
measurements separate from numerical correctness checks.

See :doc:`../concepts/configuration` for restoring global settings and configuring
spawned workers; :doc:`../data_structures/cache` describes cache ownership.

Headers and reusable configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``merge_header``, ``'no_merge'`` (default) starts with an empty header.
``'first'`` copies the first input header. ``'only_equal'`` keeps keys present
with equal values in every input header. ``'selected_keys'`` keeps keys named
in ``merge_header_keys``, taking each key's first occurrence. The output always
records ``HIERARCH astropop imcombine nimages`` and
``HIERARCH astropop imcombine method``::

    >>> image = FrameData([[10.0]], unit='adu', meta={'OBJECT': 'Example'})
    >>> preserved = imcombine([image, image], merge_header='selected_keys',
    ...                          merge_header_keys=['OBJECT'])
    >>> print("preserved.header['OBJECT']:", preserved.header['OBJECT'])
    preserved.header['OBJECT']: Example
    >>> print("preserved.header['astropop imcombine nimages']:", preserved.header['astropop imcombine nimages'])
    preserved.header['astropop imcombine nimages']: 2

Use |ImCombiner| to retain settings for several combinations. Its constructor
uses ``max_memory`` for the budget. Unlike the convenience function,
``set_sigma_clip`` defaults to ``dev_func='mad_std'``::

    >>> combiner = ImCombiner(max_memory=16 * 1024**2, dtype='float32')
    >>> combiner.set_sigma_clip(3, center_func='median', dev_func='mad_std')
    >>> combiner.set_minmax_clip(min_value=0)
    >>> combiner.set_merge_header('first')
    >>> result = combiner.combine(images, method='median')
    >>> print('result.data:', np.round(result.data, 6))
    result.data: [[3. 3.]
     [3. 3.]]

API reference: :doc:`../reference/image`.
