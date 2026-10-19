Quality flags
=============

Astropop stores pixel and measurement quality as bit fields. Each member of an
``enum.Flag`` class has a power-of-two value; an integer can contain multiple
bits at once. FrameData uses ``PixelMaskFlags`` and aperture photometry uses
``PhotometryFlags``. Do not exchange those two schemas just because both use
integer arrays.

Selecting any of several bits
----------------------------

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
    >>> mask_from_flags(flags, [PixelMaskFlags.MASKED, PixelMaskFlags.SATURATED],
    ...                 allowed_flags_class=PixelMaskFlags).tolist()
    [False, True, True]

FrameData's ``mask`` corresponds specifically to the ``MASKED`` bit. A pixel
marked only ``SATURATED`` is not automatically masked. Use ``add_flags`` to
record status and ``mask_pixels`` when an algorithm should exclude the pixel.
See :doc:`../data_structures/framedata` for storage and configuration behavior.

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

Flags and merge API
-------------------

.. automodapi:: astropop.flags
    :no-inheritance-diagram:

.. automodapi:: astropop.image._tools
    :no-inheritance-diagram:
