Carry detector-quality information through reduction
====================================================

This example executes locally under pytest with flag storage enabled (the
default). It deliberately shows where current APIs lose quality information.
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
    >>> assert raw.flags[0, 0] == bad_bits.value
    >>> assert raw.mask.tolist() == [[True, False]]
    >>> arithmetic = imarith(raw, 10 * u.adu, '-', merge_flags='or')
    >>> np.testing.assert_allclose(arithmetic.data, [[100., 110.]])
    >>> np.testing.assert_array_equal(arithmetic.flags, raw.flags)

Restore the chosen detector policy after calibration
----------------------------------------------------

The convenience calibration functions currently use ``merge_flags='no_merge'``.
Save the detector-quality map and restore it explicitly after calibration::

    >>> detector_flags = raw.flags.copy()
    >>> bias = FrameData([[10., 10.]], unit='adu')
    >>> calibrated = subtract_bias(raw, bias)
    >>> assert not calibrated.flags.any()
    >>> calibrated.flags = detector_flags
    >>> np.testing.assert_array_equal(calibrated.flags, raw.flags)
    >>> assert calibrated.mask.tolist() == [[True, False]]

This policy assumes the calibration frame has no additional defects. If it does,
combine the detector and calibration quality maps explicitly. Do not restore
only the science flags and silently discard a bad-flat or bad-bias mask.

Combine accepted samples and save
---------------------------------

The second exposure supplies a usable sample at the defect. Combination excludes
the masked first sample but does not merge its reason bits into the output::

    >>> other = FrameData([[90., 110.]], unit='adu')
    >>> combined = imcombine([calibrated, other], method='mean')
    >>> np.testing.assert_allclose(combined.data, [[90., 110.]])
    >>> assert not combined.mask.any()
    >>> assert not combined.flags.any()

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
    ...     restored = read_framedata(path)
    ...     np.testing.assert_array_equal(restored.mask, calibrated.mask)
    ...     assert restored.flags[0, 0] == (F.MASKED | F.UNSPECIFIED).value
    ...     assert not (restored.flags[0, 0] & F.SATURATED.value)

See :doc:`../tools/flags` for bit decoding and :doc:`../data_structures/framedata`
for the FITS interchange contract. If flag storage is disabled before frame
construction, these maps are not retained; choose the policy before loading.
