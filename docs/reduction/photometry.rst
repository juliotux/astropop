.. include:: ../references.txt

Photometry
==========

Start with :doc:`../tutorials/photometry` for a local detect-measure-select
workflow. Gain scaling and empty-field behavior are described below.

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
    >>> print(f'Background: {sky:.1f}; RMS: {noise:.1f}')
    Background: 100.0; RMS: 2.0
    >>> sources = daofind(image, threshold=5, background=sky, noise=noise, fwhm=4.71)
    >>> print('len(sources):', len(sources))
    len(sources): 1
    >>> print(f"Centroid: ({sources['x'][0]:.2f}, {sources['y'][0]:.2f})")
    Centroid: (64.00, 64.00)

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

For the same isolated star, compare the segmentation and combined detection
interfaces, then fit its Gaussian width. These are alternative detection paths;
they need not produce identical lists for blends or extended sources::

    >>> from astropop.photometry import segfind, starfind, median_fwhm
    >>> segments = segfind(image, threshold=5, background=sky, noise=noise, npix=5)
    >>> refined = starfind(image, threshold=5, background=sky, noise=noise)
    >>> print('Segments:', len(segments), '; refined sources:', len(refined))
    Segments: 1 ; refined sources: 1
    >>> width = median_fwhm(image, refined['x'], refined['y'], model='gaussian')
    >>> print(f'Fitted Gaussian FWHM: {width:.2f} pixels')
    Fitted Gaussian FWHM: 4.71 pixels

The fitted width describes the selected profiles, not an automatic aperture
correction. Compare it with spatial changes across the detector and repeat the
measurement as seeing changes. Inspect segment geometry when deciding whether
a candidate is a point source; detection flux and aperture flux have different
pixel selections and sky treatment.

.. _empty-detection-behavior:

When no sources are detected
-----------------------------

The wrappers do not share an empty-table return convention. ``segfind`` and
``starfind`` raise ``ValueError('No sources found')`` after Photutils emits a
``NoDetectionsWarning``. A blank field illustrates the segmentation case::

    >>> from astropop.photometry import segfind
    >>> blank = np.zeros((32, 32))
    >>> try:
    ...     empty = segfind(blank, threshold=5, background=0., noise=1.)
    ... except ValueError as error:
    ...     print(error)
    No sources found

Currently ``daofind`` does not handle Photutils returning ``None`` for no
sources: after the warning, it raises ``TypeError`` while trying to read the
result. This is a limitation of that wrapper, not an empty source table.
Inspect the warning and traceback; do not treat every ``TypeError`` as a valid
empty field, since it can also indicate an input or implementation error.

A ``len(sources)`` check cannot handle an exception raised before a table is
returned. Inspect the mask, background/noise scale, FWHM and threshold, then
choose whether to skip the exposure or rerun detection with justified settings.
No detected source does not imply zero measured flux at a known position.

Aperture fluxes and local sky
-----------------------------

Use a numerical ``r`` for a fixed circular radius. ``r='auto'`` estimates
``0.6371 * median Gaussian FWHM``. ``r_ann=(inner, outer)`` measures a local
sky annulus, ``'auto'`` chooses ``(4*r, 6*r)``, and ``None`` disables local sky
subtraction. In this example the image still contains sky, which the annulus
estimates and subtracts during measurement::

    >>> photometry = aperture_photometry(image, [64.], [64.], r=6.,
    ...                                  r_ann=(12., 18.), bkg_error=noise, gain=1.)
    >>> flux, error = photometry['flux'][0], photometry['flux_error'][0]
    >>> print(f'Flux: {flux:.0f}; relative error: {100 * error / flux:.0f}%')
    Flux: 12429; relative error: 1%

The displayed relative error is rounded; ``flux_error`` retains the computed
value. Exact errors differ slightly between supported Photutils releases
because their treatment of fractional boundary-pixel variance differs.

.. plot::
    :include-source: false
    :alt: A stellar cutout with a six-pixel source aperture and twelve-to-eighteen-pixel sky annulus, beside a radial intensity profile with the same regions shaded.
    :caption: The seeded star used above. Blue encloses the source aperture; orange bounds the local-sky annulus. The radial-bin medians show where the stellar profile approaches the background. The displayed flux is computed by aperture_photometry; radial bins are a diagnostic, not its integration method.

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle
    from astropop.photometry import background, aperture_photometry

    rng = np.random.default_rng(42)
    y, x = np.indices((128, 128))
    radius = np.hypot(x - 64, y - 64)
    pixels = 100 + rng.normal(0, 2, x.shape) + 500 * np.exp(-radius**2 / 8)
    sky, noise = background(pixels, box_size=32)
    measured = aperture_photometry(pixels, [64.], [64.], r=6., r_ann=(12., 18.),
                                   bkg_error=noise, gain=1.)
    profile = [np.median(pixels[(radius >= i) & (radius < i + 1)]) for i in range(23)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    axes[0].imshow(pixels, origin='lower', cmap='gray', vmin=94, vmax=180)
    for value, color, label in [(6, 'C0', 'Source aperture'),
                                 (12, 'C1', 'Sky annulus'), (18, 'C1', None)]:
        axes[0].add_patch(Circle((64, 64), value, fill=False, color=color,
                                 linewidth=1.5, label=label))
    axes[0].set(xlim=(42, 86), ylim=(42, 86), xlabel='x (pixel)', ylabel='y (pixel)',
                title=f"Measured flux: {measured['flux'][0]:.0f} ADU")
    axes[0].legend(loc='upper right', fontsize='small')
    axes[1].plot(np.arange(23) + 0.5, profile, 'k.-', label='Median in radial bins')
    axes[1].axhline(sky, color='0.4', linestyle='--', label='Estimated background')
    axes[1].axvspan(0, 6, color='C0', alpha=0.15, label='Aperture: r < 6')
    axes[1].axvspan(12, 18, color='C1', alpha=0.15, label='Sky: 12 < r < 18')
    axes[1].set(xlabel='Distance from source center (pixel)', ylabel='Pixel value (ADU)',
                title='Keep the sky annulus outside the source core', xlim=(0, 23))
    axes[1].legend(fontsize='small')
    axes[1].grid(alpha=0.2)

Move the annulus or mask contaminating neighbors when it includes other stars,
bad pixels or structured sky. A flat-looking profile in this isolated example
does not establish that the same aperture and sky radii suit a crowded field.

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

Measure an already background-subtracted image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a background model has already been removed, set ``r_ann=None`` to measure
without another local subtraction. The sky uncertainty remains even though its
mean level was removed; keep supplying ``bkg_error``. Here the global scalar
estimate is adequate for the example's uniform sky::

    >>> residual = image - sky
    >>> no_annulus = aperture_photometry(residual, [64.], [64.], r=6.,
    ...                                  r_ann=None, bkg_error=noise, gain=1.)
    >>> print(f'Background-subtracted flux: {no_annulus["flux"][0]:.0f}')
    Background-subtracted flux: 12419
    >>> print('Local sky:', no_annulus['bkg'][0])
    Local sky: nan

NaN in the sky columns indicates that no annulus was measured, not necessarily
an invalid source flux. A global model followed by an annulus is also a valid
strategy: the annulus then estimates the **residual** sky in the model-subtracted
image. This can correct local model offsets. It is double subtraction only
if the same estimated sky level is subtracted again, instead of estimating
the residual. The two examples need not give identical fluxes because their
sky estimators use different pixels. See the
`Photutils background-subtraction examples
<https://photutils.readthedocs.io/en/stable/user_guide/aperture.html#background-subtraction>`_
for the underlying aperture/sky calculation.

Keep the gain and error scale explicit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The wrapper's ``gain`` both divides input pixels and becomes Photutils'
effective gain. It is therefore not a drop-in replacement for a detector gain
argument that leaves ADU values unchanged. One explicit route for a single
exposure is to convert bias-subtracted ADU data and its background RMS to
electrons first, then measure with ``gain=1``. For a detector gain of two
electrons per ADU, the example's background-subtracted flux doubles::

    >>> detector_gain = 2.  # electrons per ADU, treated as exact
    >>> electron_image = residual * detector_gain
    >>> electron_noise = noise * detector_gain
    >>> in_electrons = aperture_photometry(electron_image, [64.], [64.], r=6.,
    ...                                    r_ann=None, gain=1., bkg_error=electron_noise)
    >>> ratio = in_electrons['flux'][0] / no_annulus['flux'][0]
    >>> print(f'Electron/ADU flux ratio: {ratio:.3f}')
    Electron/ADU flux ratio: 2.000

Here ``flux`` and ``flux_error`` are numerical electron values; the function
does not attach units to them. Dividing both by ``detector_gain`` returns the
ADU scale. Calibration errors and gain uncertainty require additional modeling.
The example tests the scale convention; its seeded Gaussian image is not a
realization of every term in a detector noise model.

Photutils combines a background-only error and a source Poisson term as
``sqrt(bkg_error**2 + max(source, 0) / effective_gain)``. Its
`total-error documentation
<https://photutils.readthedocs.io/en/stable/api/photutils.utils.calc_total_error.html>`_
expects background-subtracted source data. Astropop constructs that error
before its annulus subtraction. Passing sky-containing data and an RMS that
already includes sky Poisson noise can therefore count that sky term twice.
The residual-image route above separates these terms. Removing a sky model
does not remove its photon noise from ``bkg_error``. Neither route propagates
the uncertainty of the estimated sky model itself.

Recentering an aperture
^^^^^^^^^^^^^^^^^^^^^^^

Recentering is disabled by default (``recenter_limit=None``). With a numerical
limit, ``recenter_method='com'`` uses a center of mass, ``'gaussian'`` fits a
Gaussian, and ``'quadratic'`` fits a quadratic centroid. The output records
both the original positions and adopted ``x``/``y`` positions.

Compare those coordinates and inspect the flag before trusting a shifted
aperture. This candidate starts one pixel off the star; allowing only 0.1 pixel
rejects the centroid's larger move and retains the input position::

    >>> from astropop.photometry.aperture import PhotometryFlags as P
    >>> recentered = aperture_photometry(image, [65.], [64.], r=6.,
    ...                                   r_ann=(12., 18.), bkg_error=noise,
    ...                                   recenter_limit=0.1)
    >>> print('Input/adopted x:', recentered['original_x'][0], recentered['x'][0])
    Input/adopted x: 65.0 65.0
    >>> reasons = [flag.name for flag in P if int(recentered['flags'][0]) & flag.value]
    >>> print('Recentering flags:', reasons)
    Recentering flags: ['RECENTERING_FAILED']

``RECENTERING_FAILED`` means the attempted finite move exceeded the limit.
The current comparison does not reliably flag nonfinite centroid results.
Inspect the input/adopted coordinates and cutout as well, especially for
masked, blended or empty sources; an unchanged coordinate with zero flags
alone does not prove a centroid fit succeeded.

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

Masking and reason decoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This mask removes one pixel in the source core and another in the sky annulus.
The routine records both locations in the source's flags. Decode each set bit
by name so simultaneous conditions remain visible::

    >>> bad_pixels = np.zeros(image.shape, dtype=bool)
    >>> bad_pixels[64, 64] = True
    >>> bad_pixels[64, 78] = True
    >>> masked = aperture_photometry(image, [64.], [64.], r=6.,
    ...                              r_ann=(12., 18.), bkg_error=noise, mask=bad_pixels)
    >>> reasons = [flag.name for flag in P if int(masked['flags'][0]) & flag.value]
    >>> print('Measurement flags:', reasons)
    Measurement flags: ['REMOVED_PIXEL_IN_APERTURE', 'REMOVED_PIXEL_IN_ANNULUS']
    >>> print(f'Flux with the core pixel removed: {masked["flux"][0]:.0f}')
    Flux with the core pixel removed: 11928

Masked pixels are omitted, not reconstructed. In particular, the smaller flux
after removing a stellar core cannot be repaired by simply rescaling an area.
Supply a mask for contaminating neighbors in the annulus too; clipping alone
does not guarantee that a broad neighbor is excluded.

``aperture_area`` describes the unmasked source-aperture overlap used for the
sum. The current ``bkg_area`` instead reports the geometric annulus area; it
does not shrink with masking or clipping and is not an accepted-pixel count.
Do not use it to infer the number of independent background samples.

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
