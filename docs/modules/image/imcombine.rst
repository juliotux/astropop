.. include:: ../../references.txt

Image combination
=================

The :mod:`astropop.image.imcombine` module provides ``imcombine`` for a single
combination and ``ImCombiner`` for reusable rejection, storage and header
settings. Both reduce corresponding pixels along an image stack. The sections
below describe accepted samples, nominal estimators, uncertainties and the
combiner's memory strategy.

The ``imcombine`` function
--------------------------

Import the function and its configurable class::

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

    >>> # Contaminate one exposure at one pixel to compare rejection strategies.
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

.. plot::
    :include-source: false
    :alt: Nine samples at a contaminated pixel with three combination estimates, followed by histograms of residuals from the injected bias level.
    :caption: Left: the event biases the ordinary mean at one pixel. Right: residual histograms across the output images; logarithmic counts and compressed horizontal tails keep both the central distribution and the mean's single affected pixel visible.

    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib.ticker import ScalarFormatter
    from astropop.framedata import FrameData
    from astropop.image import imcombine

    rng = np.random.default_rng(42)
    pixels = rng.normal(1000, 3, (9, 48, 48))

    # Contaminate one exposure at one pixel to compare rejection strategies.
    pixels[0, 24, 24] += 600
    frames = [FrameData(image, unit='adu') for image in pixels]
    results = {
        'Mean': imcombine(frames, method='mean'),
        'Median': imcombine(frames, method='median'),
        'Clipped mean': imcombine(frames, method='mean', sigma_clip=4,
                                   sigma_dev_func='mad_std'),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    axes[0].plot(np.arange(1, 10), pixels[:, 24, 24], 'ko', label='Input samples')
    for index, (name, image) in enumerate(results.items()):
        value = image.data[24, 24]
        axes[0].axhline(value, color=f'C{index}', linestyle=['-', '--', ':'][index],
                        label=f'{name}: {value:.1f} ADU')
        axes[1].hist((image.data - 1000).ravel(), bins=np.arange(-8, 81),
                     histtype='step', color=f'C{index}', label=name, linewidth=1.5)

    axes[0].axhline(1000, color='0.5', linestyle='-.', label='Injected level: 1000 ADU')
    axes[0].set(xlabel='Exposure number', ylabel='Pixel value (ADU)',
                title='One pixel with a transient event', xticks=[1, 3, 5, 7, 9])
    axes[0].legend(fontsize='small')
    axes[1].axvline(0, color='0.5', linestyle='-.')
    axes[1].set_xscale('symlog', linthresh=4)
    axes[1].set_xticks([-8, -4, 0, 4, 20, 60])
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].set(xlabel='Residual from 1000 ADU (compressed tails)',
                ylabel='Number of pixels', yscale='log', ylim=(0.7, 3000),
                title='Residuals across the combined image')
    axes[1].legend(fontsize='small')
    for ax in axes:
        ax.grid(alpha=0.2)

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

    >>> # The default rescales an accepted-sample sum to the full input count.
    >>> normalized = imcombine(pair, method='sum', minmax_clip=(None, 20))

    >>> # Disable normalization to keep only the sum of samples that survived rejection.
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

Inspect pixels with few accepted samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here the three columns retain two, one and zero samples after rejection.
The single surviving value has zero scatter, which does not establish a
noise-free measurement::

    >>> sparse = [FrameData([[10., 20., 100.]], uncertainty=2),
    ...           FrameData([[14., 100., 100.]], uncertainty=2)]
    >>> sparse_mean = imcombine(sparse, method='mean', minmax_clip=(None, 50))

    >>> print('Mean:', sparse_mean.data.tolist())
    Mean: [[12.0, 20.0, nan]]
    >>> print('Standard uncertainty:', np.round(sparse_mean.uncertainty, 3).tolist())
    Standard uncertainty: [[1.414, 0.0, nan]]
    >>> print('Missing pixels:', sparse_mean.mask.tolist())
    Missing pixels: [[False, False, True]]
    >>> for normalize in (False, True):
    ...     summed = imcombine(sparse, method='sum', minmax_clip=(None, 50),
    ...                        sum_normalize=normalize)
    ...     print(f'Normalize={normalize}: {summed.data.tolist()}')
    Normalize=False: [[24.0, 20.0, nan]]
    Normalize=True: [[24.0, 40.0, nan]]

Empty reductions can emit NumPy runtime warnings about an empty slice or
invalid division. Inspect the output mask and missing values when these occur;
do not interpret the single-sample zero uncertainty as a replacement for them.

Normalized sums extrapolate the accepted samples to the total exposure count.
They do not recover rejected flux from variable sources or correct unequal
exposure durations. The returned frame has no accepted-count map; retain such
diagnostics separately when the observing strategy requires them.

For integer output, an all-rejected pixel is filled with zero. If flag storage
is disabled, that zero is indistinguishable from a valid zero using the output
alone::

    >>> previous_flags = conf.FRAMEDATA_DISABLE_FLAGS

    >>> try:
    ...     conf.FRAMEDATA_DISABLE_FLAGS = True
    ...     integer_inputs = [FrameData([[0, 100]], dtype='int16') for _ in range(2)]
    ...     integer_result = imcombine(integer_inputs, dtype='int16',
    ...                                minmax_clip=(None, 50))
    ...     print('Integer pixels:', integer_result.data.tolist())
    ...     print('Mask:', integer_result.mask)
    ... finally:
    ...     conf.FRAMEDATA_DISABLE_FLAGS = previous_flags
    Integer pixels: [[0, 0]]
    Mask: None

Keep flags enabled or select floating output with a NaN fill when missing
pixels must remain identifiable. This choice must precede loading the frames.

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

The storage diagram separates the arrays retained for the entire call from
those sized by the working-memory budget. Enabling disk caching changes where
the internal copies live; it does not eliminate the caller's existing frames.

.. mermaid::
    :align: center
    :alt: Caller inputs feed internal copies. A dashed working-memory cluster contains the current stack tile, validity mask and statistics scratch blocks. Tiles update the complete output; the other retained arrays lie outside the working-memory budget.
    :caption: Conceptual storage ownership during combination; box sizes are not byte measurements. The dashed area is sized using memory_limit, subject to minimum working storage. Other arrays and Python overhead contribute to total process memory.

    flowchart TD
        accTitle: Image combination storage ownership
        accDescr: Caller inputs feed internal copies. A dashed working-memory cluster contains the current stack tile, validity mask and statistics scratch blocks. Tiles update the complete output; the other retained arrays lie outside the working-memory budget.
        caller["Caller inputs<br/>Arrays, frames or filenames"] --> copies["Internal input copies<br/>RAM or explicit disk cache"]
        copies --> tile
        subgraph budget["Working-memory budget: memory_limit"]
            tile["Current output tile<br/>Stack + validity mask"] --> scratch["Statistics scratch blocks"]
            scratch -. next block / tile .-> tile
        end
        scratch -->|write tile| output["Complete output<br/>Data, uncertainty and flags when enabled"]
        style budget stroke-dasharray:5 5

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
page cache, disk space and I/O speed also matter.

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

The ``ImCombiner`` class
------------------------

``ImCombiner`` stores the options used by ``combine(image_list, method=...)``.
The constructor accepts ``dtype`` for pixel storage, ``max_memory`` for the
work-tile budget, and ``use_disk_cache``/``tmp_dir`` for private input storage.
It uses the same nominal and uncertainty estimators as ``imcombine``; creating
an instance does not load images or perform a combination.

Its configuration methods change subsequent calls to ``combine``:

* ``set_sigma_clip(sigma_limits, center_func='median', dev_func='mad_std')``
  sets one-pass rejection. A scalar threshold is symmetric, a pair supplies
  lower/upper limits, and ``None`` disables it. Unlike the wrapper function,
  the method defaults to ``'mad_std'`` rather than ``'std'``.
* ``set_minmax_clip(min_value=None, max_value=None)`` sets accepted pixel-value
  bounds. A missing bound leaves that side open; two missing bounds disable
  the filter. Boundary values are accepted.
* ``set_merge_header(strategy, keys=None)`` sets the header strategy described
  above. ``keys`` names the retained cards for ``'selected_keys'``.

Configure these methods before calling ``combine``::

    >>> combiner = ImCombiner(max_memory=16 * 1024**2, dtype='float32')

    >>> # Use a robust deviation estimate so an outlier does not inflate its own threshold.
    >>> combiner.set_sigma_clip(3, center_func='median', dev_func='mad_std')
    >>> combiner.set_minmax_clip(min_value=0)
    >>> combiner.set_merge_header('first')
    >>> result = combiner.combine(images, method='median')

    >>> print('result.data:', np.round(result.data, 6))
    result.data: [[3. 3.]
     [3. 3.]]



Input and result diagnostics
----------------------------

.. list-table:: Symptoms, checks and recovery
    :header-rows: 1
    :widths: 24 40 36

    * - Symptom
      - Minimal diagnostic
      - Recovery and contract
    * - Combination pixel is NaN or masked
      - Count accepted samples and inspect input masks, NaNs and clipping limits.
      - With no accepted samples, output is masked; missing-value fill follows
        dtype. Fix rejection/data selection, not the missing-value indicator.
        See :doc:`imcombine`.
    * - Combined integer values lack fractional precision
      - Check output dtype and expected range.
      - Temporary calculations may promote, but storage casts to the selected
        type. Choose floating storage when fractional results matter.
    * - Resident memory exceeds ``memory_limit``
      - Count caller-owned and internal images, optional arrays and outputs.
      - The limit covers work tiles, not the whole process. Use paths, smaller
        storage dtype or disk caching. See :ref:`imcombine-memory-planning`.

Reference/API
-------------

.. automodapi:: astropop.image.imcombine
    :no-inheritance-diagram:


.. currentmodule:: None
