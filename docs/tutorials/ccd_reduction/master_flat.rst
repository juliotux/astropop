Construct a normalized master flat
==================================

Subtract the master bias from each flat, remove exposure-to-exposure changes
in illumination, and combine the normalized responses. A flat is a multiplicative
sensitivity correction, unlike the additive bias.

.. tutorial-notebook:: master_flat

.. tutorial-data::

Use the raw flats and the bias master from :doc:`master_bias`, or its
reference equivalent. This stage writes ``reduced/master_flat.fits``.
See :doc:`../index` for archive layout, configuration and rerun policy.
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

Read the calibration and normalize each flat
--------------------------------------------

Select flats for the science filter and detector settings using the
:ref:`calibration compatibility rules <calibration-selection>`; the bundle's
filename pattern already selects one group. Check the master bias geometry
and units before subtracting it.

Use your preceding stage's master bias if it is available; otherwise use the
reference product included in the download::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropop.framedata import read_framedata
    >>> from astropop.image import imarith, imcombine
    >>> from astropop.image.processing import subtract_bias

    >>> output = data.parent / 'reduced'
    >>> inputs = output if (output / 'master_bias.fits').exists() else data / 'reference'
    >>> master_bias = read_framedata(inputs / 'master_bias.fits')
    >>> flat_files = sorted((data / 'raw').glob('flat_*.fits'))
    >>> normalized_flats, levels = [], []
    >>> for path in flat_files:
    ...
    ...     # Remove the additive bias before estimating each flat's illumination level.
    ...     corrected = subtract_bias(read_framedata(path), master_bias)
    ...     level = np.median(corrected.data)
    ...     if not np.isfinite(level) or level <= 0:
    ...         raise ValueError('Flat illumination must have a finite positive normalization.')
    ...     levels.append(level)
    ...
    ...     # The ADU denominator makes each normalized flat dimensionless.
    ...     normalized_flats.append(imarith(corrected, level * u.adu, '/'))

    >>> print(f'Flat levels: {min(levels):.0f} to {max(levels):.0f} ADU')
    Flat levels: 8995 to 11996 ADU

    >>> master_flat = imcombine(normalized_flats, method='median', dtype='float32')

    >>> print(f'Median response: {np.median(master_flat.data):.3f}; units: {master_flat.unit.physical_type}')
    Median response: 1.000; units: dimensionless

Bias subtraction must precede normalization. The median level is appropriate
for these star-free, smoothly illuminated flats. Inspect real flats for stars,
vignetting, shutter structure and changes in illumination before adopting the
same estimator. Normalizing only after combining would weight brighter flats
differently. The normalized flat is dimensionless. The measured scalar levels
are treated as exact in these divisions; their estimation uncertainty and the
correlations introduced by normalization are not modeled.

Inspect response and save
--------------------------

.. plot:: _ext/workflow_figures.py plot_flat
    :include-source: false
    :alt: One bias-subtracted flat in ADU and the dimensionless normalized master flat, showing the same gradient and dust shadow.
    :caption: Exposure brightness is removed; the detector response gradient and dust shadow remain so they can correct the science images.

Low-response pixels amplify noise during division. A detector-quality policy
must identify unusable regions; normalization does not repair them. The
convenience calibration functions currently clear flags, so restore your chosen
quality map explicitly as described in :doc:`../diagnostics/quality_propagation`::

    >>> output.mkdir(exist_ok=True)
    >>> master_flat.write(output / 'master_flat.fits', overwrite=True)

    >>> print('Saved: master_flat.fits')
    Saved: master_flat.fits

Continue with :doc:`science_reduction`.

.. doctest::
    :hide:

    >>> _temporary.cleanup()
