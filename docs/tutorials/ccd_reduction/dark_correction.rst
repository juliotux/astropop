Construct and apply a master dark
=================================

Build a bias-subtracted master from equal-duration dark exposures, scale it to
a science exposure and save the corrected image. This example combines
FrameData, image combination and the processing functions. The simple
``subtract_dark`` interface is described in :doc:`../../modules/image/processing`.

Prepare small example inputs::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.processing import subtract_bias, subtract_dark

    >>> science = FrameData(np.full((4, 4), 1200.), unit='adu')
    >>> bias = FrameData(np.full((4, 4), 100.), unit='adu')

Select, subtract and combine dark exposures
-------------------------------------------

Construct the dark from exposures with a common duration before scaling it.
For this small example the master bias is already available and all raw darks
were taken for ten seconds at the science detector's settings::

    >>> from astropop.image import imcombine

    >>> raw_darks = [FrameData(np.full((4, 4), value), unit='adu',
    ...                        meta={'EXPTIME': 10.}) for value in (109., 110., 111.)]
    >>> dark_times = np.array([frame.meta['EXPTIME'] for frame in raw_darks])
    >>> if not np.all(np.isfinite(dark_times) & (dark_times > 0)) or not np.all(dark_times == dark_times[0]):
    ...     raise ValueError('Select dark frames with one finite positive exposure duration.')

    >>> # Remove bias before combining, so exposure scaling affects only the dark signal.
    >>> master_dark = imcombine([subtract_bias(frame, bias) for frame in raw_darks],
    ...                         method='median')
    >>> master_dark.meta['EXPTIME'] = float(dark_times[0])
    >>> dark_subtracted = subtract_dark(subtract_bias(science, bias), master_dark,
    ...                                 dark_exposure=master_dark.meta['EXPTIME'],
    ...                                 image_exposure=20.)

    >>> print(f'Master dark: {master_dark.data[0, 0]:.1f} ADU in {dark_times[0]:.0f} s')
    Master dark: 10.0 ADU in 10 s
    >>> print(f'Science after scaled subtraction: {dark_subtracted.data[0, 0]:.1f} ADU')
    Science after scaled subtraction: 1080.0 ADU

Scaling assumes dark signal is proportional to exposure time at the selected
temperature and detector settings. A dark that still contains bias would scale
that bias too. Do not scale a nonlinear dark signal or combine different dark
durations without an explicit normalization policy. Reusing one bias also
correlates corrected inputs; the independent-error arithmetic does not track
that shared covariance.

Save the corrected science frame
--------------------------------

Keep the corrected image's unit, uncertainty and mask with the FITS product.
The temporary directory below stands in for a separate reduction-output folder::

    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from astropop.framedata import read_framedata

    >>> with TemporaryDirectory() as directory:
    ...     filename = Path(directory) / 'dark_corrected.fits'
    ...     dark_subtracted.write(filename)
    ...     saved = read_framedata(filename)
    ...     print(f'Saved pixel: {saved.data[0, 0]:.1f} {saved.unit}')
    Saved pixel: 1080.0 adu

The single exposure time in the master is a property of this selected group,
not a value inferred by ``subtract_dark``. On real observations, select matching
detector settings and temperature before constructing the master.
