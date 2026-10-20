A first offline CCD reduction
=============================

This tutorial executes under pytest using NumPy, Astropy and Astropop. It creates
small local FITS files in a temporary directory; no catalog, download, external
solver or instrument data is required. The expected calibrated image is known
exactly. This is a teaching dataset, not a detector noise simulation.

Create the observations
-----------------------

Our detector has a 100 ADU bias and a smooth sensitivity gradient. We create
three biases, three uniformly illuminated flats, and two identical science
exposures containing one Gaussian star on a constant sky. Uncertainty and flag
storage must be enabled before construction (the defaults)::

    >>> import numpy as np
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from astropy import units as u
    >>> from astropop.framedata import FrameData, read_framedata
    >>> from astropop.image import imarith, imcombine
    >>> from astropop.image.processing import subtract_bias, flat_correct
    >>> from astropop.photometry import aperture_photometry
    >>> temporary = TemporaryDirectory()
    >>> folder = Path(temporary.name)
    >>> y, x = np.indices((32, 32))
    >>> response = 0.9 + 0.2 * x / 31
    >>> expected = 20 + 500 * np.exp(-((x - 16)**2 + (y - 16)**2) / (2 * 1.2**2))
    >>> for i, offset in enumerate((-1., 0., 1.)):
    ...     FrameData(np.full((32, 32), 100 + offset), unit='adu').write(
    ...         folder / f'bias_{i}.fits')
    ...     FrameData(100 + 1000 * response + offset, unit='adu').write(
    ...         folder / f'flat_{i}.fits')
    >>> for i in range(2):
    ...     FrameData(100 + expected * response, unit='adu', uncertainty=2).write(
    ...         folder / f'science_{i}.fits')

Build the calibrations
-----------------------

Combine biases with ``method='median'``. Bias-subtract flats before combining;
normalizing by the illumination level gives a dimensionless response. In real
data estimate that level from an appropriate unmasked region, with stars,
vignetting and illumination gradients considered explicitly::

    >>> master_bias = imcombine(sorted(folder.glob('bias_*.fits')), method='median')
    >>> np.testing.assert_allclose(master_bias.data, 100)
    >>> corrected_flats = [subtract_bias(read_framedata(p), master_bias)
    ...                    for p in sorted(folder.glob('flat_*.fits'))]
    >>> master_flat = imcombine(corrected_flats, method='median')
    >>> normalized_flat = imarith(master_flat, 1000 * u.adu, '/')
    >>> assert normalized_flat.unit == u.dimensionless_unscaled
    >>> np.testing.assert_allclose(normalized_flat.data, response)

Calibrate and measure
----------------------

Read one science frame at a time, subtract the bias, and divide by the normalized
flat. Registration is unnecessary because this dataset has identical pointing.
Real shifted observations need registration before combination::

    >>> reduced = []
    >>> for p in sorted(folder.glob('science_*.fits')):
    ...     frame = flat_correct(subtract_bias(read_framedata(p), master_bias),
    ...                          normalized_flat)
    ...     np.testing.assert_allclose(frame.data, expected)
    ...     assert frame.unit == u.adu
    ...     reduced.append(frame)
    >>> combined = imcombine(reduced, method='mean')
    >>> np.testing.assert_allclose(combined.data, expected)
    >>> np.testing.assert_allclose(combined.uncertainty, 0, atol=1e-12)

The final zero uncertainty is the **scatter estimate of identical exposures**,
not propagation of the science frames' nonzero errors. Mean and median combine
estimate uncertainty from observed scatter. A propagated sum has a different
contract; see :doc:`../how_to/image_combination` and
:doc:`../concepts/uncertainties`.

Measure a circular aperture and subtract the local sky using an annulus. Here
``gain=1`` keeps the ADU numerical scale and ``bkg_error=2`` supplies a background
standard deviation in that measurement scale::

    >>> measured = aperture_photometry(combined.data, [16.], [16.], r=5.,
    ...                                  r_ann=(9., 12.), bkg_error=2., gain=1.,
    ...                                  mask=combined.mask,
    ...                                  pixel_flags=combined.flags)
    >>> total_star_flux = 2 * np.pi * 500 * 1.2**2
    >>> assert abs(measured['flux'][0] / total_star_flux - 1) < 0.02
    >>> assert measured['flux_error'][0] > 0
    >>> combined.write(folder / 'combined.fits')
    >>> measured.write(folder / 'photometry.ecsv', format='ascii.ecsv')
    >>> restored = read_framedata(folder / 'combined.fits')
    >>> np.testing.assert_allclose(restored.data, expected)
    >>> assert (folder / 'photometry.ecsv').is_file()
    >>> temporary.cleanup()

The temporary products are deleted at the end. Replace the temporary directory
with your output directory to retain them; writes refuse to overwrite existing
files by default. Aperture fluxes still need your application's photometric
calibration.

For real observations choose dark/exposure scaling, bad-pixel handling, gain,
saturation limits, flat normalization, and sky estimation for the detector.
The calibration convenience functions clear flags; follow
:doc:`../how_to/quality_flags` when preserving detector defects. Neither this
example nor its exact numerical checks establish scientific validity for a
particular camera.
