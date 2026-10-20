Detect, measure and select instrumental fluxes
==============================================

This local example runs under pytest with NumPy, Photutils and Astropop. It
creates a seeded 128-square image and writes a small ECSV measurement table in
a temporary directory. No photometric reference catalog is required.

Choose background and source scales
-----------------------------------

The sky is approximately constant at 100 ADU with 2 ADU noise. The isolated
star has Gaussian width 2 pixels. Choose background meshes larger than sources,
with enough source-free pixels, but smaller than genuine sky variations. Mask
known defects and inspect background/RMS maps on real data::

    >>> import numpy as np
    >>> from astropop.photometry import background, daofind, aperture_photometry
    >>> rng = np.random.default_rng(42)
    >>> y, x = np.indices((128, 128))
    >>> image = 100 + rng.normal(0, 2, x.shape)
    >>> image += 500 * np.exp(-((x - 64)**2 + (y - 64)**2) / 8)
    >>> sky, rms = background(image, box_size=32, filter_size=3)
    >>> assert abs(sky - 100) < 1 and 1 < rms < 3
    >>> sources = daofind(image, threshold=5, background=sky, noise=rms, fwhm=4.71)
    >>> assert len(sources) >= 1
    >>> assert np.hypot(sources['x'][0] - 64, sources['y'][0] - 64) < 1

Check detections before using their coordinates. A larger aperture collects
more stellar light and more sky noise; a local annulus must avoid the star's
wings and neighbors. Compare isolated-star fluxes across aperture radii and
inspect flags rather than assuming the automatic radius suits every field.
Do not subtract the same sky once globally and again through an annulus.

Understand the gain scale
--------------------------

The current implementation measures ``data/gain`` and leaves ``bkg_error``
unchanged. Thus ``gain=2`` halves the returned flux numerical scale; supply
background noise on that divided-data scale. The detector gain convention in
your reduction must be reconciled with this API explicitly::

    >>> unit_gain = aperture_photometry(image, [64.], [64.], r=6.,
    ...                                   r_ann=(12., 18.), gain=1., bkg_error=rms)
    >>> gain_two = aperture_photometry(image, [64.], [64.], r=6.,
    ...                                  r_ann=(12., 18.), gain=2., bkg_error=rms / 2)
    >>> np.testing.assert_allclose(gain_two['flux'], unit_gain['flux'] / 2)
    >>> assert gain_two['flux_error'][0] > 0

These flux errors include the supplied background/Poisson model, not uncertainty
in the fitted local sky, aperture correction, flat-field systematics or a
photometric zero point. Supply both ``mask`` (exclude pixels) and
``pixel_flags`` (report quality reasons) when needed.

Select and export
------------------

A conservative example keeps positive finite measurements with no flagged
conditions. Your application may accept selected flag bits after inspection;
nearby-source contamination is not automatically checked for all defined bits::

    >>> usable = (np.isfinite(unit_gain['flux']) & (unit_gain['flux'] > 0)
    ...           & (unit_gain['flags'] == 0))
    >>> assert usable[0]
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     output = Path(directory) / 'instrumental.ecsv'
    ...     unit_gain[usable].write(output, format='ascii.ecsv')
    ...     assert output.is_file()

Empty fields
-------------

Do not assume all detection wrappers return an empty table. ``segfind`` raises
``ValueError('No sources found')`` when nothing is detected. Currently
``daofind`` may emit Photutils' ``NoDetectionsWarning`` and then raise
``TypeError`` while accessing its missing source table. This is a limitation of
the wrapper; it is not evidence of a valid zero-flux measurement::

    >>> import pytest
    >>> from photutils.utils.exceptions import NoDetectionsWarning
    >>> from astropop.photometry import segfind
    >>> with pytest.warns(NoDetectionsWarning):
    ...     with pytest.raises(ValueError, match='No sources found'):
    ...         segfind(np.zeros((32, 32)), threshold=5, background=0, noise=1)
    >>> with pytest.warns(NoDetectionsWarning):
    ...     with pytest.raises(TypeError):
    ...         daofind(np.zeros((32, 32)), threshold=5, background=0, noise=1, fwhm=3)

Diagnose an empty field by checking image units, sky/noise estimates, masks,
FWHM and threshold before adjusting detection parameters. Do not catch every
exception and silently treat numerical or configuration errors as “no sources.”
See :doc:`../reduction/photometry` for estimator choices and source flags.
