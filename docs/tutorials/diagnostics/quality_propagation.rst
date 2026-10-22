Carry detector-quality information through reduction
====================================================

Keep flag storage enabled (the default) to carry detector-quality information
through reduction. This example shows where current APIs lose that information.
The application chooses a policy: here a known detector defect must remain
masked after calibration, while samples from other exposures may replace it.

Mark the defect and check arithmetic
------------------------------------

``SATURATED`` records a reason; ``MASKED`` controls exclusion. Set both when the
pixel must be omitted by combination. Arithmetic computes values even at masked
pixels and, with ``merge_flags='or'``, carries the bits forward::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropop.framedata import FrameData, PixelMaskFlags as F, read_framedata
    >>> from astropop.image import imarith, imcombine
    >>> from astropop.image.processing import subtract_bias

    >>> raw = FrameData([[110., 120.]], unit='adu')
    >>> bad_bits = F.MASKED | F.SATURATED
    >>> raw.add_flags(bad_bits, where=(0, 0))

    >>> print('raw.flags[0, 0]:', raw.flags[0, 0])
    raw.flags[0, 0]: 18
    >>> print('raw.mask.tolist():', raw.mask.tolist())
    raw.mask.tolist(): [[True, False]]

    >>> arithmetic = imarith(raw, 10 * u.adu, '-', merge_flags='or')

    >>> print('arithmetic.data:', np.round(arithmetic.data, 6))
    arithmetic.data: [[100. 110.]]
    >>> print('arithmetic.flags:', arithmetic.flags.tolist())
    arithmetic.flags: [[18, 0]]

Restore the chosen detector policy after calibration
----------------------------------------------------

The convenience calibration functions currently use ``merge_flags='no_merge'``.
Save the detector-quality map and restore it explicitly after calibration::

    >>> detector_flags = raw.flags.copy()
    >>> bias = FrameData([[10., 10.]], unit='adu')
    >>> calibrated = subtract_bias(raw, bias)

    >>> print('calibrated.flags.any():', calibrated.flags.any())
    calibrated.flags.any(): False

    >>> # Restore the saved detector defects after the helper's no-merge arithmetic.
    >>> calibrated.flags = detector_flags

    >>> print('calibrated.flags:', calibrated.flags.tolist())
    calibrated.flags: [[18, 0]]
    >>> print('calibrated.mask.tolist():', calibrated.mask.tolist())
    calibrated.mask.tolist(): [[True, False]]

This policy assumes the calibration frame has no additional defects. If it does,
combine the detector and calibration quality maps explicitly. Do not restore
only the science flags and silently discard a bad-flat or bad-bias mask.

Combine accepted samples and save
---------------------------------

The second exposure supplies a usable sample at the defect. Combination excludes
the masked first sample but does not merge its reason bits into the output::

    >>> other = FrameData([[90., 110.]], unit='adu')
    >>> combined = imcombine([calibrated, other], method='mean')

    >>> print('combined.data:', np.round(combined.data, 6))
    combined.data: [[ 90. 110.]]
    >>> print('combined.mask.any():', combined.mask.any())
    combined.mask.any(): False
    >>> print('combined.flags.any():', combined.flags.any())
    combined.flags.any(): False

If both samples are masked, the output pixel is masked. A FITS round trip
preserves the Boolean mask, not the original reason bits. Import marks a
Boolean mask as ``MASKED | UNSPECIFIED``. Save a separate
quality extension or sidecar in your application if the complete bit field is
needed::

    >>> from tempfile import TemporaryDirectory
    >>> from pathlib import Path

    >>> with TemporaryDirectory() as directory:
    ...     path = Path(directory) / 'quality.fits'
    ...     calibrated.write(path)
    ...
    ...     # FITS reload restores the Boolean mask, not the original saturation reason bit.
    ...     restored = read_framedata(path)
    ...     print('restored.mask:', restored.mask.tolist())
    ...     print('restored.flags[0, 0]:', restored.flags[0, 0])
    ...     print('restored.flags[0, 0] & F.SATURATED.value:', restored.flags[0, 0] & F.SATURATED.value)
    restored.mask: [[True, False]]
    restored.flags[0, 0]: 130
    restored.flags[0, 0] & F.SATURATED.value: 0

.. plot::
    :include-source: false
    :alt: Six pairs of pixels show nominal values and flag integers through arithmetic, convenience calibration, explicit flag restoration, combination and FITS reload. Red crosses mark Boolean masks.
    :caption: The two pixels above, followed through the actual APIs. Red crosses mark masked pixels; their values are still computed by arithmetic. Restoring the detector policy lets combination replace the masked sample with 90 ADU. FITS reload of the restored frame retains its mask but changes reason bits from 18 (MASKED | SATURATED) to 130 (MASKED | UNSPECIFIED).

    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from astropy import units as u
    from astropop.framedata import FrameData, PixelMaskFlags as F, read_framedata
    from astropop.image import imarith, imcombine
    from astropop.image.processing import subtract_bias

    raw = FrameData([[110., 120.]], unit='adu')
    raw.add_flags(F.MASKED | F.SATURATED, where=(0, 0))
    arithmetic = imarith(raw, 10 * u.adu, '-', merge_flags='or')
    cleared = subtract_bias(raw, FrameData([[10., 10.]], unit='adu'))
    restored = cleared.copy()

    # Restore the saved reason bits that the calibration helper cleared.
    restored.flags = raw.flags.copy()
    combined = imcombine([restored, FrameData([[90., 110.]], unit='adu')], method='mean')
    with TemporaryDirectory() as directory:
        filename = Path(directory) / 'quality.fits'
        restored.write(filename)
        imported = read_framedata(filename)

    stages = [(raw, 'Raw'), (arithmetic, "imarith: merge_flags='or'"),
              (cleared, 'subtract_bias: cleared flags'), (restored, 'Restore detector flags'),
              (combined, 'Combine with usable exposure'), (imported, 'FITS reload of restored frame')]
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6), layout='constrained')
    for ax, (frame, title) in zip(axes.flat, stages):
        ax.imshow(frame.data, cmap='Blues', vmin=70, vmax=150, origin='lower')
        for column in range(2):
            ax.text(column, 0, f'{frame.data[0, column]:.0f} ADU\nflags: {frame.flags[0, column]}',
                    ha='center', va='center', fontsize=10)
        masked = np.flatnonzero(frame.mask[0])
        ax.plot(masked, np.full(len(masked), 0.33), 'rx', markersize=9, markeredgewidth=2)
        ax.set(title=title, xticks=[0, 1], yticks=[], xlabel='Column x (pixel)')

The FITS panel reloads the restored single exposure, rather than the combined
image. The zero flags on the combined result do not mean the original samples
were all defect-free; accepted samples and output provenance are different
questions.

Include defects in calibration frames
--------------------------------------

For a bad pixel in the bias as well as a saturated science pixel, combine both
quality maps by bitwise OR. This example keeps all reason bits and masks both
affected pixels after subtraction::

    >>> defective_bias = FrameData([[10., 10.]], unit='adu')
    >>> defective_bias.add_flags(F.MASKED | F.BAD, where=(0, 1))
    >>> quality = np.bitwise_or(raw.flags, defective_bias.flags)
    >>> corrected = subtract_bias(raw, defective_bias)
    >>> corrected.flags = quality

    >>> print('Combined reasons:', corrected.flags.tolist())
    Combined reasons: [[18, 10]]
    >>> print('Excluded pixels:', corrected.mask.tolist())
    Excluded pixels: [[True, True]]

Apply the same policy to flat and dark defects when those calibrations are
used. Quality maps must have the same pixel geometry as the science frame;
crop or transform them consistently before combining their bits. Arithmetic
still computes the nominal values, so inspect masks before using those values.

Preserve the full flag map in a sidecar
---------------------------------------

Keep the Astropop bit field in a NumPy sidecar, alongside the FITS image whose
MASK extension stores only the Boolean exclusion map. Restore the sidecar
explicitly; ``read_framedata`` does not search for it::

    >>> with TemporaryDirectory() as directory:
    ...     path = Path(directory) / 'corrected.fits'
    ...     flags_path = Path(directory) / 'corrected.flags.npy'
    ...     corrected.write(path)
    ...     np.save(flags_path, corrected.flags, allow_pickle=False)
    ...     reloaded = read_framedata(path)
    ...     reloaded.flags = np.load(flags_path, allow_pickle=False)
    ...     print('Restored reasons:', reloaded.flags.tolist())
    ...     print('Saturated:', reloaded.mask_flags(F.SATURATED).tolist())
    ...     print('Bad:', reloaded.mask_flags(F.BAD).tolist())
    Restored reasons: [[18, 10]]
    Saturated: [[True, False]]
    Bad: [[False, True]]

Keep the pair together and record its pixel geometry, processing stage and
Astropop version in your product provenance. A same-shaped sidecar from another
exposure can be assigned successfully but describes the wrong pixels. This
recipe stores Astropop reason bits, not an arbitrary instrument DQ definition.

See :doc:`../../modules/utilities/flags` for bit decoding and :doc:`../../modules/framedata/index`
for the FITS interchange contract. If flag storage is disabled before frame
construction, these maps are not retained; choose the policy before loading.
