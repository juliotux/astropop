.. include:: ../references.txt

Building a reduction workflow
=============================

Astropop supplies reduction functions, not a pipeline scheduler or a universal
instrument recipe. Your script chooses calibration files, detector parameters,
processing order, and output names. Use :class:`~astropop.file_collection.FitsFileGroup`
to select files by header values, and call individual reduction steps explicitly.

A small calibration workflow
----------------------------

The following synthetic example builds a master bias, subtracts it, and divides
by an already normalized flat. Replace the generated frames with
``read_framedata`` calls or a file group's ``framedata()`` iterator for real data::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.imcombine import imcombine
    >>> from astropop.image.processing import subtract_bias, flat_correct
    >>> biases = [FrameData(np.full((8, 8), v), unit='adu') for v in (99., 100., 101.)]
    >>> master_bias = imcombine(biases, method='median')
    >>> science = FrameData(np.full((8, 8), 1100.), unit='adu', uncertainty=5.)
    >>> normalized_flat = FrameData(np.ones((8, 8)), uncertainty=0.01)
    >>> reduced = flat_correct(subtract_bias(science, master_bias), normalized_flat)
    >>> print(f'Reduced image: {reduced.shape}, {reduced.unit}')
    Reduced image: (8, 8), adu
    >>> print(f'Pixel range: {reduced.data.min():.1f} to {reduced.data.max():.1f} ADU')
    Pixel range: 1000.0 to 1000.0 ADU

For a detector requiring dark correction, subtract the bias from dark frames
before constructing a master dark, then supply explicit science/dark exposure
times to ``subtract_dark``. Prepare flats with the same calibration choices
and normalize them before science correction. Detector-specific effects and
overscan modeling must be handled by the application; Astropop does not infer
them from an instrument name.

Order, registration, and quality
--------------------------------

Apply the same crop to science and calibration frames. Correct detector effects
before estimating star positions or registration transforms. Register images
before combining a shifted science sequence, then pass source coordinates to
photometry. Polarimetric workflows additionally require beam pairing and
retarder-angle information; see :doc:`polarimetry`.

Track masks deliberately: arithmetic marks pixels without excluding their
values, combination rejects masked samples, and calibration convenience
functions currently clear flag bits. See :doc:`ccdprocessing` and
:doc:`imarith` before choosing how to propagate detector-quality flags.

Large datasets and reproducibility
----------------------------------

Set optional-array configuration before loading frames. Iterate over science
files rather than retaining every intermediate result. FrameData's memmap
backend and the combiner's disk cache can move storage to disk; the combination
memory budget covers work tiles, not every live input/output object.

Record software versions, configuration, calibration filenames, exposure/gain
choices, rejection settings, and registration reference alongside your products.
Write output frames explicitly, for example ``reduced.write('reduced.fits')``;
choose overwrite behavior intentionally. The notebook examples linked from the
home page illustrate complete scripts, but their instrument choices are not
library defaults.
