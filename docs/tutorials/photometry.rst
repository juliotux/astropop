Measure stars in a reduced CCD image
====================================

Use the combined image from :doc:`science_reduction` to detect stars, inspect the
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
same calibration, registration and combination steps. No stars are generated
in the visible measurement workflow.

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

Measure the original image, not the background-subtracted detection image:
local annuli subtract the sky during photometry. Subtracting a global map as
well would count the background correction twice. ``gain=1`` preserves the
input numerical flux scale. In this API data are divided by ``gain``; changing
it requires a matching change to the supplied background-noise scale.

Check and export the table
---------------------------

Inspect positions, fluxes, errors and flag reasons. This example selects positive,
finite, unflagged measurements::

    >>> usable = (np.isfinite(measured['flux']) & (measured['flux'] > 0)
    ...           & (measured['flags'] == 0))
    >>> selected = measured[usable]
    >>> len(selected)
    9
    >>> for row in selected[:3]:
    ...     print(f"Flux: {row['flux']:.0f} +/- {row['flux_error']:.0f} ADU; flags: {row['flags']}")
    Flux: 59738 +/- 297 ADU; flags: 0
    Flux: 45040 +/- 273 ADU; flags: 0
    Flux: 39937 +/- 266 ADU; flags: 0
    >>> output.mkdir(exist_ok=True)
    >>> selected.write(output / 'instrumental.ecsv', format='ascii.ecsv', overwrite=True)

The formal flux error includes the supplied background/Poisson model. It does
not include every systematic, flat-field uncertainty, registration covariance,
aperture correction or uncertainty in a photometric zero point. A small error
bar therefore does not establish an accurate calibrated magnitude.

For the supplied field, compare measured positions and fluxes against
``sources.ecsv`` only as a simulation check. On real data use isolated stars,
repeat measurements across exposures, inspect flagged cases and choose which
flags your application can accept. See :doc:`../reduction/photometry` for the
estimator contracts and source flags, and :doc:`../how_to/catalogs` for matching
a calibration catalog. Diagnose empty detections by inspecting masks, noise,
FWHM and threshold rather than treating them as zero-flux measurements.

Inspect recovery in the supplied simulation
--------------------------------------------

The injected source list provides an independent reference for this example.
Match each measured position to its nearest injected star, then display the
largest positional offset and the range of recovered flux fractions::

    >>> from astropy.table import Table
    >>> truth = Table.read(data / 'sources.ecsv')
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
here; crowded fields need more care. Continue with :doc:`astrometric_calibration`
to attach sky coordinates, or :doc:`light_curve` to follow individual exposures.

.. plot:: _ext/workflow_figures.py plot_photometry
    :include-source: False
    :alt: Stellar apertures and sky annuli on the reduced field, the sky map, and measured flux residuals relative to injected values.
    :caption: Detected stars with six-pixel apertures (cyan), sky annuli (orange), estimated background, and flux recovery against the injected stars.

.. doctest::
    :hide:

    >>> _temporary.cleanup()
