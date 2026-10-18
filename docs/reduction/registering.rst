.. include:: ../references.txt

Image Alignment and Registering
===============================

The :mod:`astropop.image.register` module estimates the relative motion between
images and resamples them onto a reference image's pixel grid. It provides
cross-correlation for translations and asterism matching for stellar fields
with translation and rotation. Registration operates on pixel coordinates;
it does not use a WCS to find the alignment.

Registering a list of frames
----------------------------

Use :func:`~astropop.image.register.register_framedata_list` with a non-empty
list of :class:`~astropop.framedata.FrameData` objects of the same two-dimensional
shape. The first frame is the reference by default. Set ``ref_image`` to another
non-negative list index to choose a different reference.

This complete example creates a moving image displaced four columns left and
three rows down from the reference:


>>> import numpy as np
>>> from astropop.framedata import FrameData
>>> from astropop.image.register import (compute_shift_list,
...                                     register_framedata_list)
>>> rng = np.random.default_rng(42)
>>> reference = np.zeros((96, 128))
>>> reference[32:60, 45:75] = rng.uniform(1, 10, (28, 30))
>>> moving = np.roll(reference, (3, -4), axis=(0, 1))
>>> frames = [FrameData(reference), FrameData(moving)]
>>> shifts = compute_shift_list(frames)
>>> np.allclose(shifts, [[0, 0], [-4, 3]])
True
>>> aligned = register_framedata_list(frames, cval=0)
>>> np.allclose(aligned[0].data, aligned[1].data)
True
>>> aligned[1].shape == moving.shape
True

The returned list preserves input order, including the reference frame.
``inplace=False`` (the default) creates new frames and leaves the originals
unchanged. Use ``inplace=True`` to modify the input frames.
:func:`~astropop.image.register.compute_shift_list` only measures shifts; it
does not resample or modify the frames. Calling it before registration is
optional: registration computes its own transforms.

Choosing an algorithm
---------------------

``algorithm='cross-correlation'`` is the default. It compares the image pixels
using phase cross-correlation and estimates translation only. Increase
``upsample_factor`` for subpixel sampling of the correlation peak; for example,
10 gives a sampling interval of one tenth of a pixel. This interval is not a
guarantee of measurement accuracy:

.. code-block:: python

    aligned = register_framedata_list(
        frames, algorithm='cross-correlation', upsample_factor=10)

``algorithm='asterism-matching'`` detects sources, sorts them by brightness,
and matches stellar triangles using ``astroalign``. It can accommodate rotation
and scale changes as well as translation. Use it when the images contain enough
common, well-separated stars:

.. code-block:: python

    aligned = register_framedata_list(
        frames, algorithm='asterism-matching', max_control_points=50,
        detection_threshold=5, detection_function='segfind')

``max_control_points`` limits the number of bright sources used in matching.
``detection_threshold`` sets the detection threshold relative to the estimated
background noise. The supported detectors are ``'segfind'`` (the default),
``'starfind'``, and ``'daofind'``. Additional keyword arguments go to the chosen
detector. Asterism matching requires usable background estimates and enough
matching sources; very small or crowded regions can fail these requirements.

Registration window
-------------------

Pass ``window=(row_slice, column_slice)`` to estimate registration using only
one rectangular region of each image. This is useful when another part of the
image contains artifacts or objects that should not determine the alignment.
The window follows NumPy indexing: rows (y) first, then columns (x), with the
stop index excluded. Negative and omitted bounds work as in NumPy, including
clipping bounds to the image extent. Each slice must have a step of one (or
omit the step), and the selected region must be non-empty.

Using the frames from the example above:


>>> window = (slice(20, 75), slice(30, 90))
>>> shifts = compute_shift_list(frames, window=window)
>>> np.allclose(shifts, [[0, 0], [-4, 3]])
True
>>> aligned = register_framedata_list(frames, window=window, cval=0)
>>> np.allclose(aligned[1].data, reference)
True

The same window is selected in every image. Choose a region large enough to
contain common features despite the expected motion. Both algorithms support
windows, and asterism source detection and background estimation also use only
the selected pixels. Cross-correlation requires real-space input when a window
is used; combining a window with ``space='fourier'`` raises ``ValueError``.

The window controls transform estimation, not the output extent. Registration
applies the resulting transform to the full moving image. Transforms are
converted back to full-image coordinates, including when asterism matching
finds a rotation. By default, ``window=None`` uses the entire image.

Shift and coordinate conventions
--------------------------------

Shifts are returned as ``[dx, dy]`` in pixels: x means columns and y means rows.
This order differs from the ``(rows, columns)`` order used for windows and NumPy
indexing. Positive x points toward increasing column indices; positive y points
toward increasing row indices.

The returned transform maps a reference-image coordinate to the corresponding
coordinate in the moving image. It is the sampling map used to construct the
aligned output. Thus a moving image displaced left by four columns and down by
three rows has a reported translation of ``[-4, 3]``. Aligning it moves its
features right by four columns and up by three rows.

For a transform containing rotation or scale, the translation is one component
of the full transform about the image origin, not the displacement of every
star. ``compute_shift_list`` returns only that translation component. Use
``compute_transform`` to obtain the full transform.

Registering individual images
-----------------------------

Create :class:`~astropop.image.register.CrossCorrelationRegister` or
:class:`~astropop.image.register.AsterismRegister` to work with one pair of
images. Algorithm options and ``window`` belong in the constructor:


>>> from astropop.image.register import CrossCorrelationRegister
>>> register = CrossCorrelationRegister(window=window, upsample_factor=10)
>>> transform = register.compute_transform(reference, moving)
>>> np.allclose(transform.translation, [-4, 3])
True
>>> image, mask, transform = register.register_image(
...     reference, moving, cval=0)
>>> np.allclose(image, reference)
True
>>> result = register.register_framedata(frames[0], frames[1], cval=0)
>>> np.allclose(result.data, image)
True

``compute_transform`` returns a scikit-image transform without changing the
arrays. ``register_image`` returns the aligned NumPy array, a boolean mask
(``True`` marks invalid pixels), and the transform. ``register_framedata``
returns a ``FrameData`` and supports ``inplace=True`` for the moving frame.

Output pixels, masks, and metadata
----------------------------------

Output images keep their original dimensions unless ``clip_output=True`` is
passed to ``register_framedata_list``. Pixels sampled from outside the moving
image are filled with ``cval`` and masked. ``cval`` can be a number, ``'median'``
(the default), or ``'mean'``. The statistics are computed over the full moving
image, even when registration uses a window.

``clip_output=True`` trims all output frames using the measured translations.
This cropping does not account for rotation or scale, so it is not a guarantee
of a fully valid common footprint for asterism matching. Check output masks
when selecting pixels for further analysis.

Both algorithms currently ignore input masks when estimating the transform.
A window excludes pixels outside its rectangle, but masked pixels inside it
can still influence the estimate. When an image is resampled, its moving-image
mask is resampled with nearest-neighbor interpolation and includes out-of-bounds
pixels. The identical-image shortcut returns the unchanged image with an empty
registration mask.

Image data are resampled with cubic interpolation. Frame uncertainties, when
present, are resampled with linear interpolation and filled with NaN outside
the input. This interpolates uncertainty values; it is not a propagation of
resampling-induced covariance. Registered frames mark invalid output pixels
with ``PixelMaskFlags.MASKED`` and ``PixelMaskFlags.OUT_OF_BOUNDS``. An existing
WCS is removed because registration does not update it.

The following metadata keys record each frame's transform. They are written
as FITS ``HIERARCH`` keywords:

.. list-table:: Registration metadata
    :header-rows: 1
    :widths: 45 55

    * - Key
      - Value
    * - ``astropop registration``
      - Algorithm name, or ``'failed'``.
    * - ``astropop registration_shift_x``
      - x translation in pixels.
    * - ``astropop registration_shift_y``
      - y translation in pixels.
    * - ``astropop registration_rot``
      - Rotation in degrees.

Handling registration failures
------------------------------

By default, an estimation or registration error is raised immediately. For
list processing, ``skip_failure=True`` logs failures and preserves the list
length and order:

* ``compute_shift_list`` returns ``[nan, nan]`` for a failed moving frame.
* ``register_framedata_list`` fills a failed frame with ``cval``, masks all its
  pixels, sets ``astropop registration`` to ``'failed'``, and sets the shift and
  rotation metadata to ``None``. Failed frames are retained in the output.

Invalid algorithm choices, incompatible frame lists, and invalid constructor
options still raise errors before per-frame processing. Check the returned
shifts, masks, or metadata before using a batch in later reductions.

Registering API
---------------

.. automodapi:: astropop.image.register
    :no-inheritance-diagram:
