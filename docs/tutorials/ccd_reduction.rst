Reduce a small CCD observing sequence
=====================================

Start from raw FITS files, build master calibrations, correct each science
exposure, register the field and combine it. The supplied observations are
synthetic, so this tutorial works offline and its results can be checked against
known inputs. The data-generation code is hidden from the reduction walkthrough.

Get the observations
---------------------

.. tutorial-data::

Extract the archive in your working directory. It contains a ``data`` directory
with eight bias frames, six flats and six science exposures in ``raw/``.
``reference/`` contains reduced products for comparison and for the standalone
photometry tutorial. ``sources.ecsv`` records the injected stars; it is for
validation, not an input to the reduction. The optional
:download:`dataset generator <../_ext/tutorial_data.py>` reproduces these files.

The simulated detector has 256 by 256 pixels, a gain of one electron per ADU,
4-electron read noise, a structured bias and a sensitivity gradient with a dust
shadow. Science exposures contain nine stars, a sloping sky, Poisson noise,
small pointing shifts and isolated bright pixel events near the image edge.
Flat illumination varies between exposures. Dark current, overscan,
nonlinearity and saturation are not simulated.

Use the extracted directory when running the examples:

.. code-block:: python

    from pathlib import Path
    data = Path('data')

.. doctest::
    :hide:

    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from tutorial_data import generate_dataset
    >>> _temporary = TemporaryDirectory()
    >>> data = Path(_temporary.name).resolve() / 'data'
    >>> generate_dataset(data)

Build the master bias and normalized flat
------------------------------------------

Keep uncertainties and flags enabled before loading frames (the default).
Select calibration frames by their observing log and FITS metadata in a real
sequence; the filenames here already separate the exposure types::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropop.framedata import read_framedata
    >>> from astropop.image import imarith, imcombine
    >>> from astropop.image.processing import subtract_bias, flat_correct
    >>> bias_files = sorted((data / 'raw').glob('bias_*.fits'))
    >>> flat_files = sorted((data / 'raw').glob('flat_*.fits'))
    >>> science_files = sorted((data / 'raw').glob('science_*.fits'))
    >>> len(bias_files), len(flat_files), len(science_files)
    (8, 6, 6)
    >>> master_bias = imcombine(bias_files, method='median', dtype='float32')
    >>> normalized_flats = []
    >>> for path in flat_files:
    ...     flat = subtract_bias(read_framedata(path), master_bias)
    ...     level = np.median(flat.data)
    ...     normalized_flats.append(imarith(flat, level * u.adu, '/'))
    >>> master_flat = imcombine(normalized_flats, method='median', dtype='float32')
    >>> print(f'Flat units: {master_flat.unit.physical_type}')
    Flat units: dimensionless
    >>> print(f'Median normalized response: {np.median(master_flat.data):.3f}')
    Median normalized response: 1.000

Bias subtraction precedes flat normalization. Normalize each flat separately
because illumination changes between exposures. The median is suitable for this
star-free simulated flat; real flats require inspection for stars, defects and
illumination structure. The master flat should be near one, with no units.

.. plot::
    :include-source: False
    :alt: Master bias showing row and column structure, alongside the normalized flat sensitivity gradient and dust shadow.
    :caption: Master bias structure and normalized sensitivity from the supplied observations.

    from tutorial_data import plot_calibrations
    plot_calibrations()

Calibrate individual science frames
-----------------------------------

Estimate each raw frame's standard uncertainty from its signal and read noise,
then subtract the bias and divide by the normalized flat::

    >>> calibrated = []
    >>> for path in science_files:
    ...     frame = read_framedata(path)
    ...     signal = np.maximum(frame.data - master_bias.data, 0)
    ...     frame.uncertainty = np.sqrt(signal + frame.meta['RDNOISE']**2)
    ...     corrected = flat_correct(subtract_bias(frame, master_bias), master_flat)
    ...     calibrated.append(corrected)

This expression uses the dataset's gain of one electron per ADU. Other gain
values need conversion of both signal and read noise into a consistent unit.
Calibration propagates the supplied errors, including the master-frame errors.
Choose detector masks and dark/exposure scaling for your camera; these data do
not require a dark subtraction. The convenience calibration functions clear
flags, so follow :doc:`../how_to/quality_flags` when carrying detector defects
through a real reduction.

Register and reject isolated events
-----------------------------------

The pointing changes between science exposures. Estimate translations from the
central field, avoiding the injected edge events, then align the full images.
The first exposure supplies the reference coordinates::

    >>> from astropop.image.register import register_framedata_list
    >>> aligned = register_framedata_list(
    ...     calibrated, ref_image=0, upsample_factor=20,
    ...     window=(slice(32, 224), slice(32, 224)))
    >>> combined = imcombine(aligned, method='mean', sigma_clip=4,
    ...                      sigma_dev_func='mad_std', dtype='float32')
    >>> print(f'Combined image: {combined.shape}, {combined.unit}, {combined.dtype}')
    Combined image: (256, 256), adu, float32

``sigma_clip=4`` rejects samples using the default median center and a
``'mad_std'`` deviation estimate before averaging. Inspect the data when
choosing a threshold: transient rejection and registration are different steps.
Interpolation also correlates neighboring pixels. The combined uncertainty is
estimated from accepted-sample scatter, rather than simply propagating the raw
frame uncertainties; see :doc:`../how_to/image_combination`.

.. plot::
    :include-source: False
    :alt: Raw science exposure, calibrated exposure and registered combined image on a common intensity scale.
    :caption: One raw exposure, its calibration, and the registered clipped mean, shown on a shared intensity scale.

    from tutorial_data import plot_reduction
    plot_reduction()

Save and inspect the products
------------------------------

Write master calibrations and the combined image, keeping metadata and present
uncertainty/flag extensions::

    >>> output = data.parent / 'reduced'
    >>> output.mkdir(exist_ok=True)
    >>> master_bias.write(output / 'master_bias.fits')
    >>> master_flat.write(output / 'master_flat.fits')
    >>> combined.write(output / 'combined.fits')
    >>> restored = read_framedata(output / 'combined.fits')
    >>> print(f'Reloaded image: {restored.shape}, {restored.unit}')
    Reloaded image: (256, 256), adu
    >>> print(f'Products written: {len(list(output.glob("*.fits")))}')
    Products written: 3

Writes refuse to overwrite existing files by default. Keep the output directory
when processing your observations. Compare sky structure, stellar shapes and
alignment across the field before accepting the stack. Continue with
:doc:`photometry` to turn the reduced image into instrumental measurements.

.. doctest::
    :hide:

    >>> _temporary.cleanup()
