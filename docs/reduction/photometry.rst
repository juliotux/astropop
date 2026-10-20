.. include:: ../references.txt

Photometry
==========

Start with :doc:`../tutorials/photometry` for a local detect-measure-select
workflow, including gain scaling and empty-field behavior.

:mod:`astropop.photometry` provides background estimation, source detection,
width estimation, and circular-aperture photometry. Pass two-dimensional
numerical arrays, rather than FrameData objects, to these routines. For a
FrameData, pass ``frame.data``, and explicitly supply its mask, flags, and the
appropriate noise model where supported. Configuration that disables FrameData
arrays does not disable the photometry routines' own error calculations.

Coordinates are zero-based ``(x, y)`` pixel centers, while images are indexed
``data[y, x]``. A ``True`` mask value excludes a pixel. Work on calibrated images
and choose detector gain/background noise in units consistent with the data.

Background and detection
------------------------

``background`` divides the image into ``box_size`` meshes and filters the
mesh estimates with ``filter_size``. ``bkg_method='mode'`` uses the
SExtractor background estimator, ``'mean'`` uses a mean, and ``'median'`` uses
a median. ``rms_method='std'`` estimates standard deviation; ``'mad_std'``
uses Gaussian-scaled median absolute deviation. ``sigma_clip`` controls mesh
clipping. The default ``global_bkg=True`` returns two scalar estimates; with
``False`` it returns background and RMS maps of the image shape::

    >>> import numpy as np
    >>> from astropop.photometry import background, daofind, aperture_photometry
    >>> rng = np.random.default_rng(42)
    >>> y, x = np.indices((128, 128))
    >>> image = 100 + rng.normal(0, 2, (128, 128))
    >>> image += 500 * np.exp(-((x - 64.)**2 + (y - 64.)**2) / (2 * 2.**2))
    >>> sky, noise = background(image, box_size=32)
    >>> assert abs(sky - 100) < 1 and 1 < noise < 3
    >>> sources = daofind(image, threshold=5, background=sky, noise=noise, fwhm=4.71)
    >>> assert len(sources) >= 1
    >>> assert np.hypot(sources['x'][0] - 64, sources['y'][0] - 64) < 1

``daofind`` uses a Gaussian kernel with supplied ``fwhm``; its threshold is a
multiple of noise above the supplied background. ``sharp_limit`` and
``round_limit`` select shapes, while ``positions`` restricts centroid searches
to supplied candidates. Spatial noise maps are reduced to a median for its
DAO threshold. ``segfind`` instead detects connected segments above the
background/noise threshold, with ``npix`` controlling minimum area.
``starfind`` combines segmentation, width estimation, and DAO detection.
All return source tables; ``x`` and ``y`` are the centroid coordinates and
``flux`` is a detection measurement, not the final aperture flux. Check for
successful detection before using source columns.

``median_fwhm`` fits source profiles using ``model='gaussian'`` or
``'moffat'``. It uses cutouts of ``box_size`` and a ``min_fwhm`` lower bound.
Use representative isolated stars, rather than saturated or blended sources.

Aperture fluxes and local sky
-----------------------------

Use a numerical ``r`` for a fixed circular radius. ``r='auto'`` estimates
``0.6371 * median Gaussian FWHM``. ``r_ann=(inner, outer)`` measures a local
sky annulus, ``'auto'`` chooses ``(4*r, 6*r)``, and ``None`` disables local sky
subtraction. Do not subtract the same background twice::

    >>> photometry = aperture_photometry(image, [64.], [64.], r=6.,
    ...                                  r_ann=(12., 18.), bkg_error=noise, gain=1.)
    >>> assert photometry['flux'][0] > 0
    >>> assert photometry['flux_error'][0] > 0

``bkg_method='mmm'`` and ``'mode'`` estimate local sky with
``3*median - 2*mean``; ``'median'`` and ``'mean'`` use those statistics directly.
``bkg_sigma_clip`` controls annulus clipping. ``bkg_error`` supplies background
standard deviation, not variance or total source uncertainty. The routine
combines background and source Poisson contributions using its gain convention.
Its implementation divides data by ``gain`` before measurement and does not
rescale the returned fluxes afterward. Thus ``gain=1`` preserves the input
numerical flux scale; for other gains the returned flux scale is that of
``data/gain``. ``bkg_error`` is passed through without that division. Supply
noise in the measurement scale and verify your detector convention explicitly.

Recentering is disabled by default (``recenter_limit=None``). With a numerical
limit, ``recenter_method='com'`` uses a center of mass, ``'gaussian'`` fits a
Gaussian, and ``'quadratic'`` fits a quadratic centroid. The output records
both the original positions and adopted ``x``/``y`` positions.

Output quality flags
--------------------

The table includes ``flux``, ``flux_error``, aperture and annulus areas, local
sky statistics, and integer ``flags``. Quality bits are defined by
:class:`~astropop.photometry.aperture.PhotometryFlags`. Decode them with bitwise
operations, rather than comparing the full flags integer to one value.
``mask`` excludes pixels from measurements; ``pixel_flags`` conveys FrameData
pixel-quality information to the resulting source flags. They serve different
purposes and should both be supplied when required. The defined
``NEARBY_SOURCES`` and ``NEARBY_SOURCES_ANNULUS`` bits are not currently set by
an automatic neighbor-contamination check. Inspect crowded apertures separately.
The returned flux error comes from the supplied background/Poisson pixel-error
model; uncertainty in the estimated annulus sky is not separately added.

Aperture fluxes are instrumental measurements. This module does not fit a
photometric zero point, extinction, color transformation, or a PSF photometry
model. Choose those calibration steps in your application and inspect quality
flags before using measurements.

Photometry API
--------------

See :doc:`../reference/photometry` for the API reference.

.. raw:: html

    <span id="astropop-photometry-package"></span>
    <span id="astropop.photometry.aperture.PhotometryFlags"></span>
    <span id="astropop.photometry.aperture.PhotometryFlags.dtype"></span>
    <span id="functions"></span>
    <span id="module-astropop.photometry"></span>
