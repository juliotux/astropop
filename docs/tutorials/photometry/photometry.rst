Measure stars in a reduced CCD image
====================================

Use the combined image from :doc:`../ccd_reduction/science_reduction` to detect stars, inspect the
background and measure local-sky-subtracted aperture fluxes. This tutorial uses
the same synthetic field and runs offline. It measures instrumental fluxes;
there is no reference-catalog zero point or transformation to calibrated magnitudes.

.. tutorial-notebook:: photometry

Load the image
---------------

.. tutorial-data::

For a standalone run, extract the archive and use its
``data/reference/combined.fits``. If you completed the CCD reduction, use your
own ``reduced/combined.fits`` instead. The reference image was produced with the
same calibration, registration and combination steps. See :doc:`../index` for
configuration and rerun policy.

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
    >>> generate_dataset(data, include_extras=False)

Read the calibrated FITS image, including its available mask and pixel flags::

    >>> import numpy as np
    >>> from astropop.framedata import read_framedata
    >>> from astropop.photometry import background, daofind, aperture_photometry

    >>> output = data.parent / 'reduced'
    >>> inputs = output if (output / 'combined.fits').exists() else data / 'reference'
    >>> image = read_framedata(inputs / 'combined.fits')

Estimate the sky and detect stars
----------------------------------

The sky varies across the field. Use background meshes larger than the stellar
profiles and inspect the resulting map. The input image retains its sky;
``background`` estimates it without modifying the pixels::

    >>> sky, rms = background(image.data, box_size=32, filter_size=3,
    ...                       mask=image.mask, global_bkg=False)

    >>> # Supply the background separately; detection handles its subtraction.
    >>> sources = daofind(image.data, threshold=5, background=sky, noise=rms,
    ...                   fwhm=4.24, mask=image.mask)

    >>> len(sources)
    9

``threshold=5`` is in units of the estimated noise. The FWHM is approximately
4.24 pixels for this dataset; estimate it from unsaturated stars in your images.
DAOfind uses the median of an input noise map for its detection threshold.
A detected source is a candidate, not an automatically usable measurement.

Measure with a local sky annulus
--------------------------------

Use a radius of six pixels and a sky annulus from nine to thirteen pixels.
These stars are isolated; crowded fields need different aperture/annulus
choices or a different measurement method::

    >>> measured = aperture_photometry(
    ...     image.data, sources['x'], sources['y'], r=6., r_ann=(9., 13.),
    ...     gain=1., bkg_error=rms, mask=image.mask, pixel_flags=image.flags)

This example measures the original image and estimates its sky in local annuli.
An alternative is to subtract the global map first and let an annulus estimate
only the residual sky. Those are different estimators, not inherently duplicate
subtractions. Subtracting the same estimated sky twice would be incorrect.
See :doc:`../../modules/photometry/index` for both paths and their error conventions.
``gain=1`` preserves the input numerical flux scale. In this API data are
divided by ``gain``; changing it also requires a matching background-noise scale.

Check and export the table
---------------------------

Inspect positions, fluxes, errors and flag reasons. This example selects positive,
finite, unflagged measurements::

    >>> # Reject flagged or nonpositive measurements before calculating flux ratios.
    >>> usable = (np.isfinite(measured['flux']) & (measured['flux'] > 0)
    ...           & (measured['flags'] == 0))
    >>> selected = measured[usable]

    >>> len(selected)
    9
    >>> for row in selected[:3]:
    ...     relative_error = 100 * row['flux_error'] / row['flux']
    ...     print(f"Flux: {row['flux']:.0f} ADU; relative error: {relative_error:.1f}%; flags: {row['flags']}")
    Flux: 59738 ADU; relative error: 0.5%; flags: 0
    Flux: 45040 ADU; relative error: 0.6%; flags: 0
    Flux: 39937 ADU; relative error: 0.7%; flags: 0

    >>> output.mkdir(exist_ok=True)
    >>> selected.meta.update({'input_set': inputs.name, 'image': 'combined.fits',
    ...                      'aperture_radius_pixels': 6., 'sky_annulus_pixels': [9., 13.],
    ...                      'gain': 1., 'selection': 'finite positive flux and flags == 0'})
    >>> selected.write(output / 'instrumental.ecsv', format='ascii.ecsv', overwrite=True)

The displayed relative errors are rounded; the saved ``flux_error`` column
retains the computed values. Exact errors differ slightly between supported
Photutils releases because their fractional boundary-pixel variance weighting
differs. This measurement deliberately illustrates the convenience wrapper's
background/Poisson model, rather than using ``image.uncertainty`` from the stack;
that array is not an argument to ``aperture_photometry``. For an ideal mean of
N equal independent exposures, a source with mean S ADU has Poisson variance
S/(N*g), where g is the detector gain in electrons per ADU. Here the routine
instead uses its single-image term S/g. Rejected samples, interpolation and
unequal accepted counts prevent a universal replacement of g by N*g; the
wrapper would also divide the nominal pixels when changing ``gain``.

The supplied RMS estimates background fluctuations in the stack. Because the
input still contains sky, the wrapper's pre-annulus Poisson calculation can
also count sky noise already present in that RMS. These formal errors are not
a calibrated uncertainty model for this clipped/interpolated stack. Measure
the individual calibrated exposures for time-series errors, or construct and
propagate an appropriate pixel-error/covariance model with an aperture API that
accepts it. The module guide explains the residual-image alternative.

The displayed errors also omit systematic, flat-field uncertainty, registration covariance,
aperture correction or uncertainty in a photometric zero point. A small error
bar therefore does not establish an accurate calibrated magnitude.

For the supplied field, compare measured positions and fluxes against
``sources.ecsv`` only as a simulation check. On real data use isolated stars,
repeat measurements across exposures, inspect flagged cases and choose which
flags your application can accept. See :doc:`../../modules/photometry/index` for the
estimator contracts and source flags, and :doc:`../../modules/catalogs/selection` for matching
a calibration catalog. Diagnose empty detections by inspecting masks, noise,
FWHM and threshold rather than treating them as zero-flux measurements.
See :ref:`empty-detection-behavior` for the wrappers' warnings and exceptions.

Inspect recovery in the supplied simulation
--------------------------------------------

The injected source list provides an independent reference for this example.
Match each measured position to its nearest injected star, then display the
largest positional offset and the range of recovered flux fractions::

    >>> from astropy.table import Table

    >>> truth = Table.read(data / 'sources.ecsv')

    >>> # Match measured centroids to simulation truth only for this recovery diagnostic.
    >>> distance2 = ((selected['x'][:, None] - truth['x'])**2
    ...              + (selected['y'][:, None] - truth['y'])**2)
    >>> nearest = np.argmin(distance2, axis=1)
    >>> offset = np.sqrt(np.min(distance2, axis=1))
    >>> recovery = 100 * selected['flux'] / truth['flux'][nearest]

    >>> print(f'Largest position offset: {np.max(offset):.2f} pixels')
    Largest position offset: 0.04 pixels
    >>> print(f'Recovered flux: {np.min(recovery):.1f}% to {np.max(recovery):.1f}%')
    Recovered flux: 99.2% to 100.6%

The aperture contains most of the stellar light, but finite aperture size,
background estimation, interpolation and noise prevent exact recovery. These
numbers describe this seeded dataset; they are not accuracy guarantees for
other observations. The isolated stars make nearest-neighbor matching suitable
here; crowded fields need more care. Continue with :doc:`../astrometry_catalogs/astrometric_calibration`
to attach sky coordinates, or :doc:`light_curve` to follow individual exposures.

.. plot:: _ext/workflow_figures.py plot_photometry
    :include-source: false
    :alt: Stellar apertures and sky annuli on the reduced field, the sky map, and measured flux residuals relative to injected values.
    :caption: Detected stars with six-pixel apertures (cyan), sky annuli (orange), estimated background, and flux recovery against the injected stars.

.. doctest::
    :hide:

    >>> _temporary.cleanup()
