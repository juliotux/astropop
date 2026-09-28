.. include:: ../references.txt

FrameData Container
===================

|FrameData| stores a two-dimensional astronomical image together with its
physical unit, optional uncertainties, pixel flags, metadata, and WCS. It also
supports disk-backed arrays for reductions that need to limit memory use.
Use ``read_framedata`` to convert FITS images or |CCDData| objects into this
container; the constructor accepts image arrays.

Creating and copying frames
---------------------------

Provide a two-dimensional array or nested sequence. Data are stored as
``float64`` by default; use ``dtype`` to choose another floating-point type.
Integer storage is not supported. A scalar uncertainty is expanded to the
image shape::

    import numpy as np
    from astropop.framedata import FrameData, PixelMaskFlags, read_framedata

    frame = FrameData([[10, 20], [30, 40]], unit='adu', uncertainty=0.5,
                      dtype='float32', meta={'OBJECT': 'Example'})
    assert frame.shape == (2, 2)
    assert frame.size == 4
    assert frame.dtype == np.dtype('float32')
    np.testing.assert_array_equal(frame.uncertainty, np.full((2, 2), 0.5))

A |Quantity| supplies its own unit. A conflicting explicit ``unit`` raises an
error. Without either, the data are dimensionless. Access the numerical array
through ``frame.data`` and its unit through ``frame.unit``.

Construction can share a supplied floating-point array. Use ``copy()`` for an
independent frame, or ``astype()`` to copy while changing floating-point dtype.
Copies include metadata and any arrays enabled by the current configuration::

    duplicate = frame.copy()
    duplicate.data[0, 0] = 99
    assert frame.data[0, 0] == 10
    converted = frame.astype('float64')
    assert converted.dtype == np.dtype('float64')

``read_framedata(frame)`` returns the same object. Use
``read_framedata(frame, copy=True)`` to request a copy. ``check_framedata`` is an
alias of ``read_framedata``.

Arithmetic and statistics
^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``imarith`` for arithmetic with units and uncertainty propagation::

    from astropop.image.imarith import imarith

    scaled = imarith(frame, 2, '*')
    np.testing.assert_allclose(scaled.data, frame.data * 2)
    np.testing.assert_allclose(scaled.uncertainty, frame.uncertainty * 2)
    assert scaled.unit == frame.unit

Direct changes to ``frame.data`` do not propagate uncertainties.
``mean()``, ``median()``, ``std()``, ``min()``, and ``max()`` return quantities
when a physical unit is stored, and numerical values otherwise. ``statistics()`` returns these values in a dictionary.
These methods use all data pixels; they do not exclude masked pixels or weight
by uncertainty. To exclude masked pixels, use a NaN-aware NumPy function on
``frame.get_masked_data()`` and attach ``frame.unit`` to the result if needed.

Metadata and WCS
----------------

Pass a dictionary or FITS header as ``meta`` (or its constructor alias
``header``, but not both). ``frame.meta`` and ``frame.header`` expose the same
FITS header and support normal keyword access::

    frame.header['EXPTIME'] = (30.0, 'Exposure time in seconds')
    assert frame.meta['EXPTIME'] == 30.0
    frame.history = 'Created example image'
    frame.history = ['Checked units', 'Checked uncertainty']
    frame.comment = 'Demonstration frame'
    assert frame.history[-1] == 'Checked uncertainty'

Structural FITS keywords are removed when metadata are loaded and regenerated
on FITS output. HISTORY and COMMENT cards are held separately in
``frame.history`` and ``frame.comment``. Assigning a string or list to these
properties appends entries; it does not replace previous entries. FITS output
includes both lists.

WCS information is extracted from an input header into ``frame.wcs``. You may
instead pass an ``astropy.wcs.WCS`` object as ``wcs`` when constructing the
frame. Supplying WCS both in the header and explicitly raises an error.
``frame.wcs`` accepts a WCS object or ``None``; FITS output serializes it back
into the header.

Masks and uncertainties
-----------------------

Uncertainties are standard deviations in the same unit as the data. Supply a
scalar or an array matching the image shape. An absent uncertainty is ``None``;
assigning ``frame.uncertainty = None`` removes it. ``get_uncertainty()`` returns
a copy of the uncertainty array, or ``None`` when absent.

Pixel flags are a ``uint8`` array, initially zero unless flag storage is
disabled. ``PixelMaskFlags`` defines INTERPOLATED, MASKED (also called REMOVED),
DEAD, BAD, SATURATED, COSMIC_RAY, OUT_OF_BOUNDS, and UNSPECIFIED bits. Combine
flags with ``|``. A reason flag alone does not mask a pixel: include MASKED to
exclude it from operations that respect the mask::

    frame.add_flags(PixelMaskFlags.SATURATED | PixelMaskFlags.MASKED,
                    (0, 1))
    assert frame.mask[0, 1]
    assert frame.mask_flags(PixelMaskFlags.SATURATED)[0, 1]
    frame.mask_pixels((1, 0))
    masked_data = frame.get_masked_data()
    assert np.isnan(masked_data[0, 1])
    assert frame.data[0, 1] == 20

``mask_flags()`` selects pixels with any of the requested bits. ``frame.mask``
is a derived Boolean array selecting MASKED pixels; modify flags or use
``mask_pixels()`` to change the stored mask. ``get_masked_data()`` returns a
copy with masked pixels replaced by NaN, or by a specified ``fill_value``.

Constructor ``mask`` and ``flags`` arrays must match the image shape. True
values in an input mask add MASKED and UNSPECIFIED bits to the supplied flags.
When flags are absent, both ``frame.flags`` and ``frame.mask`` are ``None``.

Optional storage and propagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For reductions that do not need uncertainties or masks, configure
``AstropopConfig`` before loading frames. All three switches default to
``False``:

.. list-table:: Memory and propagation settings
    :header-rows: 1

    * - Setting
      - Effect when True
    * - ``FRAMEDATA_DISABLE_UNCERTAINTY``
      - Ignore uncertainty inputs and assignments, omit them from copies and
        newly read frames, and skip uncertainty calculations in image arithmetic
        and combination.
    * - ``FRAMEDATA_DISABLE_FLAGS``
      - Ignore mask and flag inputs and assignments. New frames and copies
        have no flags or mask arrays. Masking methods do not store flags.
    * - ``IMARITH_SKIP_UNCERTAINTY``
      - Skip uncertainty propagation in ``imarith`` and uncertainty estimation
        in ``imcombine``. The result has ``uncertainty=None``; input frames keep
        their stored uncertainties.

For example, temporarily disable uncertainty storage::

    import numpy as np
    from astropop.config import AstropopConfig
    from astropop.framedata import FrameData
    from astropop.image.imarith import imarith

    previous = AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY
    try:
        AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = True
        minimal = FrameData(np.ones((100, 100)), uncertainty=0.5)
        assert minimal.uncertainty is None
        result = imarith(minimal, 2, '*')
        assert result.uncertainty is None
    finally:
        AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = previous

These are process-wide class attributes, not per-frame settings. Changing them
is not thread-local and does not remove arrays already held by existing frames.
New assignments and copies use the current settings. Assigning an uncertainty
or flags while their storage is disabled clears any previously stored array.
Ignored inputs are not validated or converted.

Disabling uncertainty propagation preserves unit conversion and nominal data
operations. Combination still performs configured rejection and excludes stored
masked pixels. Disabling flag storage discards masks on newly loaded frames and
copies, so those masks cannot exclude pixels from a subsequent reduction.
Internal rejection masks are still used while combining, even when output flag
storage is disabled.

FITS output omits missing uncertainty and mask extensions. CCDData conversion
uses ``None`` for these fields. ``get_uncertainty(return_none=False)`` remains an
explicit request for a zero-filled array when no uncertainty is stored; it
allocates that array even with uncertainty storage disabled.

Memory mapping
--------------

In-memory frames do not create cache directories by default. To store arrays
on disk, supply an explicit ``cache_folder`` either at construction or when
calling ``enable_memmap``::

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        cached = FrameData(np.ones((100, 100)), cache_folder=directory,
                           use_memmap_backend=True)
        cached.disable_memmap()  # load arrays into memory and remove cache files

Calling ``enable_memmap()`` without a previously configured directory raises
``ValueError``. Only present arrays get cache files; disabled uncertainties and
flags do not allocate files. Copies use unique filenames in the same directory
and own their files independently. Existing files are never silently overwritten.
Cleanup removes owned cache files and preserves unrelated files and existing
user directories. Cache files are temporary working storage, not a persistent
image format; use FITS for saved results.

.. _Data IO:

Reading and writing images
---------------------------

``read_framedata`` accepts FITS filenames (including path objects), image HDUs,
HDU lists, |CCDData|, NumPy arrays, |Quantity|, and two-dimensional
``astropop.math.physical.QFloat`` objects. Arrays must be two-dimensional.
Additional constructor options, such as ``dtype`` and ``cache_folder``, can be
passed when loading a new frame.

FITS files and HDUs
^^^^^^^^^^^^^^^^^^^^^^

``write()`` saves a FITS file, and ``to_hdu()`` creates an in-memory HDU list.
The primary HDU contains the data, metadata, and WCS. By default, the unit is
stored in BUNIT, standard deviations in UNCERT, and the Boolean mask in MASK.
Missing uncertainty or mask arrays produce no corresponding extension::

    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        filename = Path(directory) / 'example.fits'
        frame.write(filename)
        restored = read_framedata(filename)
        np.testing.assert_allclose(restored.data, frame.data)
        np.testing.assert_allclose(restored.uncertainty, frame.uncertainty)
        np.testing.assert_array_equal(restored.mask, frame.mask)
        assert restored.unit == frame.unit
        assert restored.header['OBJECT'] == 'Example'

    with frame.to_hdu() as hdus:
        assert hdus[0].header['BUNIT'] == 'adu'
        assert 'UNCERT' in hdus
        assert 'MASK' in hdus

Existing files are protected unless ``overwrite=True`` is passed to
``write()``. Use ``hdu_uncertainty``, ``hdu_mask``, and ``unit_key`` to customize
extension names and the unit keyword on both reading and writing. On output,
setting an extension name to ``None`` omits that extension. The reader assumes
that the uncertainty extension contains standard deviations.

The FITS reader uses the first HDU containing image data. To read a particular
image from a multi-image file, pass that image HDU directly; pass associated
uncertainties and masks separately if required. If the unit keyword is absent,
``unit`` can supply the unit. A conflicting header unit and explicit unit raise
an error. Output permits Astropy unit strings by default; set
``no_fits_standard_units=False`` to require FITS-standard unit formatting.

FITS conversion preserves the Boolean mask, but does not serialize the full
pixel flag array or the reasons each pixel was masked. It is not a lossless
round trip for individual flag bits.

CCDData conversion
^^^^^^^^^^^^^^^^^^

Convert explicitly with ``to_ccddata()`` and ``read_framedata()``::

    ccd = frame.to_ccddata()
    restored_ccd = read_framedata(ccd)
    np.testing.assert_allclose(restored_ccd.data, frame.data)
    np.testing.assert_allclose(restored_ccd.uncertainty, frame.uncertainty)
    np.testing.assert_array_equal(restored_ccd.mask, frame.mask)

CCDData output uses ``StdDevUncertainty`` when uncertainties are present and
``None`` otherwise. Import also accepts CCDData variance and inverse-variance
uncertainties, converting them to standard deviations. Masks are transferred
as Boolean arrays; individual FrameData flag bits are not preserved. Dedicated
FrameData history and comment lists are not transferred by ``to_ccddata()``.
The optional-storage settings above also apply when importing CCDData or FITS.

FrameData API
-------------

.. automodapi:: astropop.framedata
    :no-inheritance-diagram:
