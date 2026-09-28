.. include:: ../references.txt

FramaData Container
===================

|FrameData| is a special container to store important data of astronomical images, just like |CCDData| instances. It is designed to handle data, uncertainties, masks, phyisical units, metadata, memmapping, and other things. However, it is not fully compatible with |CCDData| due to important design differences that are needed to Astropop.

FrameData Usage
---------------

.. note::
    There is no way to create |FrameData| directly from |CCDData| or |HDUList|. Please, see `Data IO`_ for data interchanging.

.. TODO:: Usage

Metadata
--------

.. TODO:: Metadata and header handling

Masks and Uncertainties
-----------------------

Uncertainties are stored as standard deviations in the same unit as the data.
An absent uncertainty is returned as ``None``. Pixel flags are stored in a
``uint8`` array; ``frame.mask`` selects pixels carrying ``PixelMaskFlags.MASKED``.
When no flags are stored, both ``frame.flags`` and ``frame.mask`` are ``None``.

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
        frame = FrameData(np.ones((100, 100)), uncertainty=0.5)
        assert frame.uncertainty is None
        result = imarith(frame, 2, '*')
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
^^^^^^^^^^^^^^

In-memory frames do not create cache directories by default. To store arrays
on disk, supply an explicit ``cache_folder`` either at construction or when
calling ``enable_memmap``::

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        frame = FrameData(np.ones((100, 100)), cache_folder=directory,
                          use_memmap_backend=True)
        frame.disable_memmap()  # load arrays into memory and remove cache files

Calling ``enable_memmap()`` without a previously configured directory raises
``ValueError``. Only present arrays get cache files; disabled uncertainties and
flags do not allocate files. Copies use unique filenames in the same directory
and own their files independently. Existing files are never silently overwritten.
Cleanup removes owned cache files and preserves unrelated files and existing
user directories. Cache files are temporary working storage, not a persistent
image format; use FITS for saved results.

.. _Data IO:

Data IO
-------

.. TODO:: Data IO (FITS, |CCDData|, |HDUList|, HDUs, etc.)

FrameData API
---------------

.. automodapi:: astropop.framedata
    :no-inheritance-diagram:
