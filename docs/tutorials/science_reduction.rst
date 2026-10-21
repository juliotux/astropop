Calibrate, register and combine science exposures
=================================================

Use the master calibrations on each exposure before estimating pointing shifts.
Save individual calibrated frames as well as the stack: the stack is useful for
deep photometry, while individual frames retain time variability.

.. tutorial-notebook:: science_reduction

.. tutorial-data::

Use the raw science exposures and the bias/flat masters from the preceding
stages, or their reference equivalents. Save corrected exposures and the
registered stack in ``reduced``.
See :doc:`index` for archive layout, configuration and rerun policy.
Run from the extracted bundle root when following the HTML examples::

    >>> from pathlib import Path

.. code-block:: python

    data = Path('data')

.. doctest::
    :hide:

    >>> from tempfile import TemporaryDirectory
    >>> from tutorial_data import generate_dataset
    >>> _temporary = TemporaryDirectory()
    >>> data = Path(_temporary.name).resolve() / 'data'
    >>> generate_dataset(data, include_extras=False)

Calibrate individual exposures
-------------------------------

Use your master calibrations when both are present, or the supplied reference
pair for a standalone run. These data have a gain of one electron per ADU, so
signal and read-noise variance can be combined numerically in the same scale::

    >>> import numpy as np
    >>> from astropy.table import Table
    >>> from astropop.framedata import read_framedata
    >>> from astropop.image import imcombine
    >>> from astropop.image.processing import subtract_bias, flat_correct
    >>> from astropop.image.register import register_framedata_list
    >>> output = data.parent / 'reduced'
    >>> own_masters = all((output / name).exists() for name in ['master_bias.fits', 'master_flat.fits'])
    >>> inputs = output if own_masters else data / 'reference'
    >>> master_bias = read_framedata(inputs / 'master_bias.fits')
    >>> master_flat = read_framedata(inputs / 'master_flat.fits')
    >>> science_files = sorted((data / 'raw').glob('science_*.fits'))
    >>> output.mkdir(exist_ok=True)
    >>> calibrated = []
    >>> for index, path in enumerate(science_files):
    ...     frame = read_framedata(path)
    ...     signal = np.maximum(frame.data - master_bias.data, 0)
    ...     frame.uncertainty = np.sqrt(signal + frame.meta['RDNOISE']**2)
    ...     detector_flags = frame.flags.copy()
    ...     corrected = flat_correct(subtract_bias(frame, master_bias), master_flat)
    ...     corrected.flags = detector_flags
    ...     corrected.write(output / f'calibrated_{index:02d}.fits', overwrite=True)
    ...     calibrated.append(corrected)
    >>> print(f'Calibrated exposures: {len(calibrated)}; units: {calibrated[0].unit}')
    Calibrated exposures: 6; units: adu

For another gain, convert both terms to a consistent unit first. This estimate
omits the uncertainty of the bias-subtracted signal inside the Poisson term;
calibration then propagates the supplied image and master-frame uncertainties.
The flag restoration assumes the masters add no additional defects. Combine
quality maps explicitly when they do. These simulated exposures need no dark
subtraction or overscan correction.

Register using the central field
--------------------------------

The edge contains injected bright events. Estimate translations from a central
window, then apply those transforms to the full images. Frame zero defines the
output coordinates::

    >>> aligned = register_framedata_list(calibrated, ref_image=0, upsample_factor=20,
    ...                                   window=(slice(32, 224), slice(32, 224)))
    >>> shifts = Table({'frame': np.arange(len(aligned)),
    ...                 'dx': [f.meta['astropop registration_shift_x'] for f in aligned],
    ...                 'dy': [f.meta['astropop registration_shift_y'] for f in aligned]})
    >>> print(f'Last reference-to-moving shift: ({shifts["dx"][-1]:.2f}, {shifts["dy"][-1]:.2f}) pixels')
    Last reference-to-moving shift: (2.00, -1.25) pixels
    >>> shifts.write(output / 'shifts.ecsv', overwrite=True)
    >>> combined = imcombine(aligned, method='mean', sigma_clip=4,
    ...                      sigma_dev_func='mad_std', dtype='float32')
    >>> combined.meta['EXPTIME'] = 30.
    >>> combined.write(output / 'combined.fits', overwrite=True)
    >>> print(f'Stack: {combined.shape}, {combined.dtype}, {combined.unit}; exposure scale: {combined.meta["EXPTIME"]:.0f} s')
    Stack: (256, 256), float32, adu; exposure scale: 30 s

Reported shifts map reference coordinates to moving-image coordinates; subtract
them to correct a source position. The registration routine already applies the
inverse transform when aligning pixels. A central window does not correct
optical distortion. Inspect stars across the field as in
:doc:`../how_to/registration_diagnostics`.

.. plot:: _ext/workflow_figures.py plot_science
    :include-source: False
    :alt: Raw exposure, bias/flat-corrected exposure and registered stack using a common intensity stretch.
    :caption: The same field before and after calibration and combination. Inspect background structure, stellar shapes and edge behavior, not only the appearance of the brightest star.

Interpret the stack and its errors
-----------------------------------

``sigma_clip=4`` uses one rejection pass with a median center and ``'mad_std'``
deviation here. Inspect its effect before applying the threshold to another
sequence. Mean and median combination estimate uncertainty from accepted-sample
scatter rather than propagating individual input errors. Interpolation also
correlates neighboring pixels, and shared master-frame errors do not average
away independently. See :doc:`../how_to/image_combination` for the estimators.

The mean remains on a single 30-second exposure's count scale, not a 180-second
sum. The EXPTIME saved above records that convention for subsequent instrumental
magnitudes; it is not a claim about the effective noise-equivalent exposure.
Continue with :doc:`photometry` for the stack or :doc:`light_curve` for a separate
time series.

.. doctest::
    :hide:

    >>> _temporary.cleanup()
