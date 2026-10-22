Quality flags
=============

Astropop stores pixel and measurement quality as bit fields. Each member of an
``enum.Flag`` class has a power-of-two value; an integer can contain multiple
bits at once. FrameData uses ``PixelMaskFlags`` and aperture photometry uses
``PhotometryFlags``. Do not exchange those two schemas just because both use
integer arrays.

Selecting any of several bits
-----------------------------

``mask_from_flags`` returns a Boolean array that is true wherever any requested
bit occurs. Supply flag members, not raw integer bit numbers. The optional
``allowed_flags_class`` rejects members of another flag class::

    >>> import numpy as np
    >>> from astropop.flags import mask_from_flags
    >>> from astropop.framedata import PixelMaskFlags

    >>> flags = np.array([0, PixelMaskFlags.MASKED.value,
    ...                   (PixelMaskFlags.MASKED | PixelMaskFlags.SATURATED).value])

    >>> mask_from_flags(flags, PixelMaskFlags.SATURATED).tolist()
    [False, False, True]

    >>> # The list selects pixels carrying either of these bits; both are not required.
    >>> mask_from_flags(flags, [PixelMaskFlags.MASKED, PixelMaskFlags.SATURATED],
    ...                 allowed_flags_class=PixelMaskFlags).tolist()
    [False, True, True]

FrameData's ``mask`` corresponds specifically to the ``MASKED`` bit. A pixel
marked only ``SATURATED`` is not automatically masked. Use ``add_flags`` to
record status and ``mask_pixels`` when an algorithm should exclude the pixel.
See :doc:`../framedata/index` for storage and configuration behavior.

Combining flags
---------------

Image arithmetic accepts ``merge_flags='or'`` to combine all bits, ``'and'``
to keep only common bits, and ``'no_merge'`` for zero output flags. Missing
flags count as zero. Those masks mark the arithmetic result; they do not remove
operand values from arithmetic. Image combination instead excludes samples
with ``MASKED`` and marks output pixels that have no accepted samples. Its
output is not a bitwise accumulation of every input quality bit.

``merge_flag`` and ``merge_header`` expose the underlying merge operations for
custom workflows. ``merge_header`` uses ``'only_equal'`` for keys equal in all
headers, ``'first'`` for the first header, ``'selected_keys'`` for explicitly
named keys by first occurrence, or ``'no_merge'`` for an empty header. Always
specify the intended method when calling these lower-level helpers.

For a custom array operation, inspect the lower-level flag strategies directly.
The example uses two-bit integer arrays so each result is visible::

    >>> from astropop.image._tools import merge_flag, merge_header

    >>> first = np.array([1, 2], dtype='uint8')
    >>> second = np.array([2, 2], dtype='uint8')

    >>> for method in ('or', 'and', 'no_merge'):
    ...     print(method, merge_flag(first, second, method=method).tolist())
    or [3, 2]
    and [0, 2]
    no_merge [0, 0]

Choose header merging independently of flag merging. ``'selected_keys'``
requires ``selected_keys``; the first occurrence supplies each selected value.
Here the filter agrees, but exposure time differs::

    >>> from astropy.io.fits import Header

    >>> first_header = Header({'FILTER': 'R', 'EXPTIME': 10})
    >>> second_header = Header({'FILTER': 'R', 'EXPTIME': 20})

    >>> for method in ('only_equal', 'first', 'selected_keys', 'no_merge'):
    ...     merged = merge_header(first_header, second_header, method=method,
    ...                           selected_keys=['EXPTIME'])
    ...     print(method, dict(merged))
    only_equal {'FILTER': 'R'}
    first {'FILTER': 'R', 'EXPTIME': 10}
    selected_keys {'EXPTIME': 10}
    no_merge {}

These operations return merged arrays/headers; they do not infer an exposure
policy or update an image's physical calibration. The arithmetic wrapper
applies the chosen strategies to its result.




Input and result diagnostics
----------------------------

.. list-table:: Symptoms, checks and recovery
    :header-rows: 1
    :widths: 24 40 36

    * - Symptom
      - Minimal diagnostic
      - Recovery and contract
    * - Mask or reason bits disappear
      - Inspect ``flags`` and ``mask`` before and after every operation and
        check whether flag storage is disabled.
      - Calibration clears flags; FITS interchange keeps Boolean masks.
        Adopt an explicit policy using :doc:`../../tutorials/diagnostics/quality_propagation`.

Reference/API
-------------

.. automodapi:: astropop.flags
    :no-inheritance-diagram:

.. automodapi:: astropop.image._tools
    :no-inheritance-diagram:


.. currentmodule:: None
