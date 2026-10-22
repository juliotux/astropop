.. include:: ../../references.txt

.. _framedata-container:

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
``float64`` by default; use ``dtype`` to choose another integer or
floating-point type. Integer frames also store their uncertainties as integers,
truncating fractional values. A scalar uncertainty is expanded to the image
shape::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData, PixelMaskFlags, read_framedata

    >>> frame = FrameData([[10, 20], [30, 40]], unit='adu', uncertainty=0.5,
    ...                   dtype='float32', meta={'OBJECT': 'Example'})

    >>> print('frame.shape:', frame.shape)
    frame.shape: (2, 2)
    >>> print('frame.size:', frame.size)
    frame.size: 4
    >>> print('frame.dtype:', frame.dtype)
    frame.dtype: float32
    >>> print('frame.uncertainty:', np.round(frame.uncertainty, 6))
    frame.uncertainty: [[0.5 0.5]
     [0.5 0.5]]

A |Quantity| supplies its own unit. A conflicting explicit ``unit`` raises an
error. Without either, the data are dimensionless. Access the numerical array
through ``frame.data`` and its unit through ``frame.unit``. ``shape``, ``size``,
and ``dtype`` describe the stored data; flags always use ``uint8`` independently
of the pixel dtype.

Construction can share a supplied array of the selected dtype. Use ``copy()``
for an independent frame, or ``astype()`` to copy while changing dtype.
Copies include metadata and any arrays enabled by the current configuration::

    >>> duplicate = frame.copy()

    >>> # Change the copy to demonstrate that its pixels are independent.
    >>> duplicate.data[0, 0] = 99

    >>> print('frame.data[0, 0]:', frame.data[0, 0])
    frame.data[0, 0]: 10.0

    >>> converted = frame.astype('float64')

    >>> print('converted.dtype:', converted.dtype)
    converted.dtype: float64

Integer frames require finite data and uncertainties. Masks remain separate
from the pixels. ``get_masked_data()`` returns a floating-point copy when a
nonfinite fill such as the default NaN is requested; a finite integer fill
keeps the integer dtype.

``read_framedata(frame)`` returns the same object and ignores constructor
options, including ``dtype``. Use ``read_framedata(frame, copy=True)`` to request
a copy, or ``read_framedata(frame, copy=True, dtype='float32')`` to convert the
copy. ``check_framedata`` is an alias of ``read_framedata``.

The ``data``, ``uncertainty``, and ``flags`` properties expose the stored arrays
directly. Editing one of these arrays changes the frame. In contrast,
``get_uncertainty()`` and ``get_masked_data()`` return independent arrays.
``copy()`` also copies the header, WCS, history, and comments; it preserves
``origin_filename`` as descriptive metadata rather than opening that file.

Arithmetic and statistics
^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``imarith`` for arithmetic with units and uncertainty propagation::

    >>> from astropop.image.imarith import imarith

    >>> scaled = imarith(frame, 2, '*')

    >>> print('scaled.data:', np.round(scaled.data, 6))
    scaled.data: [[20. 40.]
     [60. 80.]]
    >>> print('scaled.uncertainty:', np.round(scaled.uncertainty, 6))
    scaled.uncertainty: [[1. 1.]
     [1. 1.]]
    >>> print('scaled.unit:', scaled.unit)
    scaled.unit: adu

FrameData does not overload Python arithmetic operators. Direct changes to
``frame.data`` do not propagate uncertainties. See
:doc:`../image/imarith` for arithmetic and :doc:`../image/imcombine` for
rejection and combination examples.
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

    >>> frame.header['EXPTIME'] = (30.0, 'Exposure time in seconds')

    >>> print("frame.meta['EXPTIME']:", frame.meta['EXPTIME'])
    frame.meta['EXPTIME']: 30.0

    >>> frame.history = 'Created example image'
    >>> frame.history = ['Checked units', 'Checked uncertainty']
    >>> frame.comment = 'Demonstration frame'

    >>> print('frame.history[-1]:', frame.history[-1])
    frame.history[-1]: Checked uncertainty

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
a copy in the data dtype, or ``None`` when absent. A scalar uncertainty is
expanded across the image. Convert uncertainty quantities to the data unit
before assigning their numerical values::

    >>> from astropy import units as u

    >>> # Convert uncertainty values to the data unit before passing the plain array.
    >>> errors = (np.full((2, 2), 10) * u.cm).to_value(u.m)
    >>> measured = FrameData(np.ones((2, 2)), unit='m', uncertainty=errors)

    >>> print('measured.get_uncertainty():', np.round(measured.get_uncertainty(), 6))
    measured.get_uncertainty(): [[0.1 0.1]
     [0.1 0.1]]

    >>> no_errors = FrameData(np.ones((2, 2)))

    >>> print('no_errors.get_uncertainty():', no_errors.get_uncertainty())
    no_errors.get_uncertainty(): None
    >>> print('no_errors.get_uncertainty(False):', np.round(no_errors.get_uncertainty(False), 6))
    no_errors.get_uncertainty(False): [[0. 0.]
     [0. 0.]]

Absence means the uncertainty is not stored or calculated; it is not a
measurement of zero error.

Pixel flags are a ``uint8`` array, initially zero unless flag storage is
disabled. ``PixelMaskFlags`` defines INTERPOLATED, MASKED (also called REMOVED),
DEAD, BAD, SATURATED, COSMIC_RAY, OUT_OF_BOUNDS, and UNSPECIFIED bits. Combine
flags with ``|``. A reason flag alone does not mask a pixel: include MASKED to
exclude it from operations that respect the mask::

    >>> frame.add_flags(PixelMaskFlags.SATURATED | PixelMaskFlags.MASKED,
    ...                 (0, 1))

    >>> print('frame.mask[0, 1]:', frame.mask[0, 1])
    frame.mask[0, 1]: True
    >>> print('frame.mask_flags(PixelMaskFlags.SATURATED)[0, 1]:', frame.mask_flags(PixelMaskFlags.SATURATED)[0, 1])
    frame.mask_flags(PixelMaskFlags.SATURATED)[0, 1]: True

    >>> frame.mask_pixels((1, 0))

    >>> # Insert NaNs in a separate array; the stored pixel values stay intact.
    >>> masked_data = frame.get_masked_data()

    >>> print('Masked pixel:', masked_data[0, 1])
    Masked pixel: nan
    >>> print('frame.data[0, 1]:', frame.data[0, 1])
    frame.data[0, 1]: 20.0

``mask_flags()`` selects pixels with any of the requested bits. ``frame.mask``
is a derived Boolean array selecting MASKED pixels; modify flags or use
``mask_pixels()`` to change the stored mask. ``get_masked_data()`` returns a
copy with masked pixels replaced by NaN, or by a specified ``fill_value``.

Constructor ``mask`` and ``flags`` arrays must match the image shape. True
values in an input mask add MASKED and UNSPECIFIED bits to the supplied flags.
Assigning ``frame.flags = None`` removes the flag array; both ``frame.flags``
and ``frame.mask`` then return ``None``. Calling ``mask_pixels()`` or
``add_flags()`` allocates flags again when storage is enabled.

.. _optional-frame-arrays:

Optional storage and propagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For reductions that do not need uncertainties or masks, you must set the
``AstropopConfig`` class attributes **before loading or creating frames**,
including calls to ``read_framedata`` and ``FrameData``. Do not instantiate the
configuration class. Setting these options only before arithmetic or combination
does not recover memory already allocated for existing frames. Both switches
default to ``False``:

.. list-table:: Memory and propagation settings
    :header-rows: 1

    * - Setting
      - Effect when True
    * - ``FRAMEDATA_DISABLE_UNCERTAINTY``
      - Ignore uncertainty inputs and assignments, omit them from copies and
        newly read frames, return ``None`` from uncertainty getters, and skip
        uncertainty calculations in image arithmetic and combination.
    * - ``FRAMEDATA_DISABLE_FLAGS``
      - Ignore mask and flag inputs and assignments. New frames and copies
        have no flags or mask arrays. Flag and mask getters return ``None``;
        masking methods do not store flags.

For example, set the option before creating any frames for the reduction::

    >>> import numpy as np
    >>> from astropop.config import AstropopConfig
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.imarith import imarith

    >>> previous = AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY

    >>> try:
    ...     AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = True
    ...     minimal = FrameData(np.ones((100, 100)), uncertainty=0.5)
    ...     print('minimal.uncertainty:', minimal.uncertainty)
    ...     result = imarith(minimal, 2, '*')
    ...     print('result.uncertainty:', result.uncertainty)
    ... finally:
    ...     AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = previous
    minimal.uncertainty: None
    result.uncertainty: None

These are process-wide class attributes, not per-frame settings. Changing them
is not thread-local. The settings are checked on each getter or uncertainty
calculation call. Disabling storage hides arrays already held by existing
frames without deleting them: the corresponding properties and mask methods
return ``None``. Re-enabling storage exposes those arrays again. This also
applies to memory-mapped arrays, whose files remain available while hidden.
New assignments and copies use the current settings. Assigning an uncertainty
or flags while their storage is disabled clears any previously stored array,
so re-enabling storage cannot restore it. Ignored inputs are not validated or
converted.

Disabling uncertainty propagation preserves unit conversion and nominal data
operations. Combination still performs configured rejection and excludes stored
masked pixels. Disabling flag storage hides masks on existing frames and
discards them on newly loaded frames and copies, so those masks cannot exclude
pixels from a subsequent reduction.
Internal rejection masks are still used while combining, even when output flag
storage is disabled.

In-place arithmetic replaces the first frame's uncertainty with ``None`` when
uncertainty storage is disabled, because that frame is also the result. The
same setting also omits uncertainty interpolation during registration and
slicing during trimming.

Combination avoids uncertainty working copies, chunks, result arrays and
error-only standard-deviation calculations when uncertainty storage is disabled.
Input adapters also skip uncertainty conversions that would allocate arrays.
Rejection uses its own data buffers and continues to operate. With propagation
enabled, rejected samples are excluded from the summed uncertainties as well as
from the nominal data.

For a frame containing :math:`P` pixels, stored data require
``P * frame.dtype.itemsize`` bytes, flags require another ``P`` bytes when
present, and uncertainties require ``frame.uncertainty.nbytes`` when present.
This describes array storage only: masks returned by getters, temporary
calculations, metadata, and cache bookkeeping require additional memory.
Configure storage before reading images to avoid allocating arrays that will
only be hidden later.

FITS output omits missing or disabled uncertainty and mask extensions. CCDData
conversion uses ``None`` for these fields. ``get_uncertainty(return_none=False)``
remains an explicit request for a zero-filled array when the uncertainty is
missing or disabled; it allocates that array even if an existing uncertainty
array is hidden by the configuration.

Memory mapping
--------------

Frames retain the existing cache-directory policy: the default is a temporary
folder under the Astropy cache. ``enable_memmap()`` uses that configured folder;
no explicit directory is required. To choose the location yourself::

    >>> from tempfile import TemporaryDirectory

    >>> with TemporaryDirectory() as directory:
    ...     cached = FrameData(np.ones((100, 100)), cache_folder=directory,
    ...                        use_memmap_backend=True)
    ...     cached.disable_memmap()  # load present arrays and remove their files

Use a dedicated cache directory. The current cache manager removes the directory
on cleanup, including any other files it contains. Cache ownership policy is
unchanged by the optional-array feature.

Only present arrays create mapped files. Removing uncertainty or flags closes
and deletes that array's file without reading it back into memory. Disabling
memmapping loads present arrays into RAM. Copies use separate cache directories,
with an additional unique suffix when the usual copy directory already exists.
Cache files are temporary working storage; use FITS for saved results.

.. _Data IO:

Reading and writing images
---------------------------

``read_framedata`` accepts FITS filenames (including path objects), image HDUs,
HDU lists, |CCDData|, NumPy arrays, |Quantity|, and two-dimensional
``astropop.math.physical.QFloat`` objects. Arrays must be two-dimensional.
Additional constructor options, such as ``dtype`` and ``cache_folder``, can be
passed when loading a new frame.

For example, a QFloat image supplies nominal values, standard uncertainties,
and units together::

    >>> from astropop.math.physical import QFloat

    >>> quantity_image = QFloat(np.ones((2, 2)), np.full((2, 2), 0.2), 'adu')
    >>> adapted = read_framedata(quantity_image, dtype='float32')

    >>> print('adapted.dtype:', adapted.dtype)
    adapted.dtype: float32
    >>> print('adapted.uncertainty:', np.round(adapted.uncertainty, 6))
    adapted.uncertainty: [[0.2 0.2]
     [0.2 0.2]]
    >>> print('adapted.unit:', adapted.unit)
    adapted.unit: adu

Masked-array or Quantity masks are not inferred by the FrameData constructor.
Pass a separate ``mask=`` array when loading numerical values that carry a
mask; FITS and CCDData adapters transfer their Boolean masks automatically.

FITS files and HDUs
^^^^^^^^^^^^^^^^^^^^^^

``write()`` saves a FITS file, and ``to_hdu()`` creates an in-memory HDU list.
The primary HDU contains the data, metadata, and WCS. By default, the unit is
stored in BUNIT, standard deviations in UNCERT, and the Boolean mask in MASK.
Missing uncertainty or mask arrays produce no corresponding extension::

    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory

    >>> with TemporaryDirectory() as directory:
    ...     filename = Path(directory) / 'example.fits'
    ...     frame.write(filename)
    ...     restored = read_framedata(filename)
    ...     print('restored.data:', np.round(restored.data, 6))
    ...     print('restored.uncertainty:', np.round(restored.uncertainty, 6))
    ...     print('restored.mask:', restored.mask.tolist())
    ...     print('restored.unit:', restored.unit)
    ...     print("restored.header['OBJECT']:", restored.header['OBJECT'])
    restored.data: [[10. 20.]
     [30. 40.]]
    restored.uncertainty: [[0.5 0.5]
     [0.5 0.5]]
    restored.mask: [[False, True], [True, False]]
    restored.unit: adu
    restored.header['OBJECT']: Example

    >>> with frame.to_hdu() as hdus:
    ...     print("hdus[0].header['BUNIT']:", hdus[0].header['BUNIT'])
    ...     print('Extensions:', [hdu.name for hdu in hdus])
    hdus[0].header['BUNIT']: adu
    Extensions: ['PRIMARY', 'UNCERT', 'MASK']

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

Select science, variance and quality extensions explicitly
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Instrument files can contain several images, variance rather than standard
deviation, and a quality map whose nonzero values mark defects. Select the
science HDU itself and convert the companion arrays according to that file's
documented convention. Naming a variance extension ``hdu_uncertainty`` would
load its values as standard deviations without taking a square root::

    >>> from astropy.io import fits

    >>> hdus = fits.HDUList([
    ...     fits.PrimaryHDU(),
    ...     fits.ImageHDU(np.zeros((2, 2)), name='PREVIEW'),
    ...     fits.ImageHDU(np.array([[10., 20.], [30., 40.]]), name='SCI'),
    ...     fits.ImageHDU(np.full((2, 2), 4.), name='VAR'),
    ...     fits.ImageHDU(np.array([[0, 4], [0, 0]], dtype='uint8'), name='DQ'),
    ... ])
    >>> hdus['SCI'].header['BUNIT'] = 'adu'

    >>> with hdus:
    ...     variance = hdus['VAR'].data
    ...     if np.any(~np.isfinite(variance)) or np.any(variance < 0):
    ...         raise ValueError('Variance must be finite and nonnegative for this import.')
    ...     selected = read_framedata(hdus['SCI'], uncertainty=np.sqrt(variance),
    ...                               mask=hdus['DQ'].data != 0, dtype='float32')
    ...     print('Selected extension:', hdus['SCI'].name)
    Selected extension: SCI

    >>> print('Science:', selected.data.tolist())
    Science: [[10.0, 20.0], [30.0, 40.0]]
    >>> print('Standard uncertainty:', selected.uncertainty.tolist())
    Standard uncertainty: [[2.0, 2.0], [2.0, 2.0]]
    >>> print('Mask:', selected.mask.tolist())
    Mask: [[False, True], [False, False]]

The variance here is in ADU squared, matching the science data's ADU unit.
Real DQ definitions may require selecting particular bits instead of masking
every nonzero value. External DQ bit numbers are not automatically Astropop
``PixelMaskFlags`` values; keep the original quality map if its reasons matter.
See :doc:`../../tutorials/diagnostics/quality_propagation` for saving a complete Astropop flag map.

CCDData conversion
^^^^^^^^^^^^^^^^^^

Convert explicitly with ``to_ccddata()`` and ``read_framedata()``::

    >>> ccd = frame.to_ccddata()
    >>> restored_ccd = read_framedata(ccd)

    >>> print('restored_ccd.data:', np.round(restored_ccd.data, 6))
    restored_ccd.data: [[10. 20.]
     [30. 40.]]
    >>> print('restored_ccd.uncertainty:', np.round(restored_ccd.uncertainty, 6))
    restored_ccd.uncertainty: [[0.5 0.5]
     [0.5 0.5]]
    >>> print('restored_ccd.mask:', restored_ccd.mask.tolist())
    restored_ccd.mask: [[False, True], [True, False]]

CCDData output uses ``StdDevUncertainty`` when uncertainties are present and
``None`` otherwise. Import also accepts CCDData variance and inverse-variance
uncertainties, converting them to standard deviations. Masks are transferred
as Boolean arrays; individual FrameData flag bits are not preserved. Dedicated
FrameData history and comment lists are not transferred by ``to_ccddata()``.
The optional-storage settings above also apply when importing CCDData or FITS.




Input and result diagnostics
----------------------------

.. list-table:: Symptoms, checks and recovery
    :header-rows: 1
    :widths: 24 40 36

    * - Symptom
      - Minimal diagnostic
      - Recovery and contract
    * - FITS input has missing/wrong data or uncertainty
      - Inspect HDU names, extension shapes and unit header cards.
      - Select extensions explicitly and verify standard-deviation convention.
        See :doc:`index`.

Reference/API
-------------

.. automodapi:: astropop.framedata
    :no-inheritance-diagram:


.. currentmodule:: None

Related topics
--------------

.. toctree::
    :maxdepth: 1

    uncertainties
