Check registration at the center and edge
=========================================

A recovered central translation does not prove that every source is aligned.
Distortion, changing PSFs, saturation, and a bright source entering/leaving an
edge can make different regions favor different shifts. A window limits the
pixels used to estimate a transform; the transform still applies to the full
image. It does not remove optical distortion.

This example uses three images and the **middle image** as reference.
Two central stars drift by a known translation. A much brighter stationary edge
feature deliberately conflicts with them. This is an adversarial diagnostic,
not a physical model of an instrument's distortion::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.register import register_framedata_list

    >>> y, x = np.indices((128, 128))
    >>> positions = np.array([[55., 50.], [75., 70.], [10., 10.]])
    >>> drift = np.array([[-2., -1.], [0., 0.], [2., 1.]])  # (dx, dy)
    >>> frames = []
    >>> for dx, dy in drift:
    ...     image = np.zeros(x.shape)
    ...     for sx, sy in positions[:2] + [dx, dy]:
    ...         image += 100 * np.exp(-((x - sx)**2 + (y - sy)**2) / 2)
    ...
    ...     # Keep a bright edge feature fixed while the two central stars drift.
    ...     image += 1e6 * np.exp(-((x - 10)**2 + (y - 10)**2) / 2)
    ...     frames.append(FrameData(image))

    >>> full = register_framedata_list(frames, ref_image=1)
    >>> windowed = register_framedata_list(
    ...     frames, ref_image=1, window=(slice(35, 90), slice(35, 90)))
    >>> def measured_shifts(registered):
    ...     return np.array([[f.meta['astropop registration_shift_x'],
    ...                       f.meta['astropop registration_shift_y']]
    ...                      for f in registered])

    >>> full_shift = measured_shifts(full)
    >>> window_shift = measured_shifts(windowed)

    >>> print('full_shift:', np.round(full_shift, 1) + 0.0)  # Display signed zeros as zero.
    full_shift: [[0. 0.]
     [0. 0.]
     [0. 0.]]
    >>> print('window_shift:', np.round(window_shift, 6))
    window_shift: [[-2. -1.]
     [ 0.  0.]
     [ 2.  1.]]

Reported shifts describe the reference-to-moving transform in ``(x, y)`` order.
For a pure translation, **subtract** those shifts from moving-image coordinates
to express them on the reference grid. Image arrays remain indexed ``[y, x]``::

    >>> observed = np.broadcast_to(positions, (3, 3, 2)).copy()
    >>> observed[:, :2] += drift[:, None, :]
    >>> residual_full = observed - full_shift[:, None, :] - positions
    >>> residual_window = observed - window_shift[:, None, :] - positions
    >>> central_offsets = np.linalg.norm(residual_window[:, :2], axis=2)

    >>> print('Largest central residual per frame:', np.round(central_offsets.max(axis=1), 2))
    Largest central residual per frame: [0. 0. 0.]
    >>> print(f'Central residual with full image: {np.linalg.norm(residual_full[0, 0]):.2f} pixels')
    Central residual with full image: 2.24 pixels
    >>> print(f'Edge residual with window: {np.linalg.norm(residual_window[0, 2]):.2f} pixels')
    Edge residual with window: 2.24 pixels

Windowing recovers both central stars but leaves an edge residual here. In real
observations centroid several isolated stars spanning the field before and
after registration, then plot residual ``dx`` and ``dy`` against radius and frame
number. For a middle reference, expected drift is relative to that middle frame,
not the first frame. For rotation/scale use the full transform rather than
subtracting a translation alone.

Inspect which region the transform aligns
-----------------------------------------

.. plot::
    :include-source: false
    :alt: Reference image with a central window, and central and edge residuals for full-image and windowed registration against the middle frame.
    :caption: The cyan box selects the faint central stars. Full-image correlation follows the bright stationary feature, while windowed correlation follows the central drift. Both residual curves are relative to frame 1.

    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    from astropop.framedata import FrameData
    from astropop.image.register import register_framedata_list

    y, x = np.indices((128, 128))
    positions = np.array([[55., 50.], [75., 70.], [10., 10.]])
    drift = np.array([[-2., -1.], [0., 0.], [2., 1.]])
    frames = []
    for dx, dy in drift:
        image = np.zeros(x.shape)
        for sx, sy in positions[:2] + [dx, dy]:
            image += 100 * np.exp(-((x - sx)**2 + (y - sy)**2) / 2)

        # Keep a bright edge feature fixed while the two central stars drift.
        image += 1e6 * np.exp(-((x - 10)**2 + (y - 10)**2) / 2)
        frames.append(FrameData(image))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), layout='constrained')
    axes[0].imshow(np.maximum(frames[1].data, 1), origin='lower', cmap='magma',
                    norm=LogNorm(vmin=1, vmax=1e6))
    axes[0].add_patch(Rectangle((34.5, 34.5), 55, 55, fill=False,
                                edgecolor='cyan', linewidth=1.5))
    axes[0].set(title='Middle reference and window', xlabel='x (pixel)', ylabel='y (pixel)')
    observed = np.broadcast_to(positions, (3, 3, 2)).copy()
    observed[:, :2] += drift[:, None, :]
    for name, window, color in [('Full image', None, 'C0'),
                                ('Central window', (slice(35, 90), slice(35, 90)), 'C1')]:
        registered = register_framedata_list(frames, ref_image=1, window=window)
        shifts = np.array([[frame.meta['astropop registration_shift_x'],
                            frame.meta['astropop registration_shift_y']]
                           for frame in registered])
        residual = np.linalg.norm(observed - shifts[:, None, :] - positions, axis=2)
        for ax, values in zip(axes[1:], [residual[:, :2].max(axis=1), residual[:, 2]]):
            ax.plot(np.arange(3), values, 'o-', color=color, label=name)

    for ax, title in zip(axes[1:], ['Largest central-star residual', 'Edge-feature residual']):
        ax.set(title=title, xlabel='Frame index (reference = 1)',
                ylabel='Position residual (pixel)', xticks=[0, 1, 2], ylim=(-0.1, 2.6))
        ax.grid(alpha=0.2)

    axes[1].legend(fontsize='small')

The reference image uses a logarithmic intensity stretch to reveal the faint
central stars; correlation still operates on the original pixel values.
The reference frame has zero residual for both methods by construction. Compare
the other frames: the full-image estimate leaves the central stars displaced,
while the central window leaves the conflicting edge feature displaced. This
illustrates why one good central alignment is insufficient to claim alignment
across a distorted field. The stationary feature here is deliberately artificial;
these curves are not a measurement of any camera's optical distortion.

The example raises on failed registration. If using ``skip_failure=True``, inspect
the per-frame registration metadata and masks; do not convert failed shifts to
zero or connect them through a residual plot as successful estimates. Review
sources crossing the estimation window, saturation and edge masks before
changing algorithms. See :doc:`../../modules/image/register` for coordinate conventions,
interpolation and failure policy.

Register rotation and scale, then correct coordinates
-----------------------------------------------------

Asterism matching estimates a similarity transform from common stars. Build a
small field whose moving-image positions have a known scale change, rotation
and translation. Render both fields from those positions so the simulation does
not introduce an extra image-resampling step::

    >>> from skimage.transform import SimilarityTransform
    >>> from astropop.image.register import AsterismRegister

    >>> reference_xy = np.array([
    ...     [40, 40], [90, 45], [160, 35], [215, 70], [55, 110], [120, 115],
    ...     [185, 130], [35, 190], [100, 210], [170, 195], [220, 220], [200, 45]], dtype=float)
    >>> injected = SimilarityTransform(scale=1.02, rotation=np.deg2rad(2),
    ...                                translation=(-3, 4))
    >>> moving_xy = injected(reference_xy)
    >>> yy, xx = np.indices((256, 256))
    >>> fields = []
    >>> for positions_xy in (reference_xy, moving_xy):
    ...     pixels = np.full(xx.shape, 10.)
    ...     for index, (sx, sy) in enumerate(positions_xy):
    ...         pixels += (1000 + 100 * index) * np.exp(-((xx - sx)**2 + (yy - sy)**2) / 8)
    ...     fields.append(pixels)

    >>> matcher = AsterismRegister(detection_threshold=5, max_control_points=20)
    >>> transform = matcher.compute_transform(fields[0], fields[1])

    >>> print(f'Scale: {transform.scale:.3f}; rotation: {np.rad2deg(transform.rotation):.2f} deg')
    Scale: 1.020; rotation: 2.00 deg
    >>> print('Translation (pixels):', np.round(transform.translation, 2))
    Translation (pixels): [-3.  4.]

    >>> corrected_xy = transform.inverse(moving_xy)
    >>> offset = np.linalg.norm(corrected_xy - reference_xy, axis=1)
    >>> translation_only = moving_xy - transform.translation
    >>> wrong_offset = np.linalg.norm(translation_only - reference_xy, axis=1)

    >>> print(f'Largest full-transform residual: {offset.max():.3f} pixels')
    Largest full-transform residual: 0.000 pixels
    >>> print(f'Largest translation-only residual: {wrong_offset.max():.1f} pixels')
    Largest translation-only residual: 12.6 pixels

The reported transform maps reference positions into the moving image. Its
``inverse`` therefore maps measured moving-image coordinates back to the
reference grid. Subtracting ``transform.translation`` alone leaves rotation
and scale around the image origin uncorrected. The displayed zero residual is
rounded and belongs to this noiseless isolated-star simulation, not a measured
accuracy bound for real images. With a registration window, use the returned
full-image transform directly; do not apply another crop-origin offset.

.. plot::
    :include-source: false
    :alt: Translation-only correction leaves position-dependent residual vectors across the field, while the full inverse mapping overlays the reference positions.
    :caption: Geometry of the injected 1.02 scale, 2-degree rotation and (-3, 4) translation. These panels use the known transform to illustrate the coordinate rule; the fitted transform is assessed numerically above. Arrows are residuals in pixel units, without magnification.

    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.transform import SimilarityTransform

    reference = np.array([[40, 40], [90, 45], [160, 35], [215, 70], [55, 110],
                          [120, 115], [185, 130], [35, 190], [100, 210],
                          [170, 195], [220, 220], [200, 45]], dtype=float)
    injected = SimilarityTransform(scale=1.02, rotation=np.deg2rad(2), translation=(-3, 4))
    observed = injected(reference)

    # Removing translation alone leaves the injected rotation and scale uncorrected.
    residual = observed - injected.translation - reference
    corrected = injected.inverse(observed)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), layout='constrained')
    axes[0].scatter(*reference.T, color='0.4', s=20)
    axes[0].quiver(*reference.T, *residual.T, angles='xy', scale_units='xy', scale=1)
    axes[0].set_title('Translation-only residuals')
    axes[1].scatter(*reference.T, s=55, facecolors='none', edgecolors='k', label='Reference')
    axes[1].scatter(*corrected.T, s=15, label='Full inverse correction')
    axes[1].set_title('Full coordinate correction')
    axes[1].legend(fontsize='small')
    for ax in axes:
        ax.set(xlim=(0, 256), ylim=(0, 256), xlabel='x (pixel)', ylabel='y (pixel)', aspect='equal')
        ax.grid(alpha=0.2)

Select a valid footprint and handle a failed exposure
-----------------------------------------------------

The coordinate transform also tells the image resampler where to read moving
pixels. It can require samples outside that image. Keep flags enabled and use
the output mask to select usable pixels instead of assuming a rectangular crop
removes every invalid sample::

    >>> reference_frame, moving_frame = [FrameData(pixels) for pixels in fields]
    >>> registered = matcher.register_framedata(reference_frame, moving_frame, cval=0)
    >>> common_valid = ~reference_frame.mask & ~registered.mask

    >>> print(f'Common valid pixels: {100 * common_valid.mean():.1f}%')
    Common valid pixels: 92.0%

``common_valid`` is a pixel selection on the reference grid, not a new image
coordinate system. Use it when measuring or combining those registered pixels.
``clip_output=True`` uses translations for cropping and cannot guarantee a
rotation/scale footprint. Interpolation and registration uncertainties still
need their own error treatment even inside the valid area.

A blank moving frame has no stars to match. With ``skip_failure=True`` the
failed exposure remains in the returned list and is fully masked; it must not
be interpreted as a successfully registered zero-valued exposure::

    >>> blank_frame = FrameData(np.zeros(fields[0].shape))
    >>> batch = register_framedata_list(
    ...     [reference_frame, blank_frame], algorithm='asterism-matching',
    ...     skip_failure=True, cval=0, detection_threshold=5, max_control_points=20)

    >>> print('Output frames:', len(batch))
    Output frames: 2
    >>> print('Failed-frame status:', batch[1].meta['astropop registration'])
    Failed-frame status: failed
    >>> print('Usable pixels in failed frame:', np.count_nonzero(~batch[1].mask))
    Usable pixels in failed frame: 0

The detector emits a no-source warning and registration logs the failure.
Keep input indices or filenames alongside the results when rejecting failed
frames. A fully failed list cannot provide an image stack or a common useful
footprint; report that condition rather than replacing failed shifts by zero.
