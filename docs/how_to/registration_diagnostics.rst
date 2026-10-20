Check registration at the center and edge
=========================================

A recovered central translation does not prove that every source is aligned.
Distortion, changing PSFs, saturation, and a bright source entering/leaving an
edge can make different regions favor different shifts. A window limits the
pixels used to estimate a transform; the transform still applies to the full
image. It does not remove optical distortion.

This local pytest example uses three images and the **middle image** as reference.
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
    >>> np.testing.assert_allclose(full_shift, 0, atol=0.1)
    >>> np.testing.assert_allclose(window_shift, drift, atol=0.1)

Reported shifts describe the reference-to-moving transform in ``(x, y)`` order.
For a pure translation, **subtract** those shifts from moving-image coordinates
to express them on the reference grid. Image arrays remain indexed ``[y, x]``::

    >>> observed = np.broadcast_to(positions, (3, 3, 2)).copy()
    >>> observed[:, :2] += drift[:, None, :]
    >>> residual_full = observed - full_shift[:, None, :] - positions
    >>> residual_window = observed - window_shift[:, None, :] - positions
    >>> np.testing.assert_allclose(residual_window[:, :2], 0, atol=0.1)
    >>> assert np.linalg.norm(residual_full[0, 0]) > 2
    >>> assert np.linalg.norm(residual_window[0, 2]) > 2

Windowing recovers both central stars but leaves an edge residual here. In real
observations centroid several isolated stars spanning the field before and
after registration, then plot residual ``dx`` and ``dy`` against radius and frame
number. For a middle reference, expected drift is relative to that middle frame,
not the first frame. For rotation/scale use the full transform rather than
subtracting a translation alone.

The example raises on failed registration. If using ``skip_failure=True``, inspect
the per-frame registration metadata and masks; do not convert failed shifts to
zero or connect them through a residual plot as successful estimates. Review
sources crossing the estimation window, saturation and edge masks before
changing algorithms. See :doc:`../reduction/registering` for coordinate conventions,
interpolation and failure policy.
