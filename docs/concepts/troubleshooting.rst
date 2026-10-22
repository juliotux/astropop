Troubleshoot a reduction
========================

Start with the smallest failing frame or source list. Record software versions,
configuration, input shape/dtype/unit and the exact exception. Separate a
scientific limitation from a reproducible implementation error; do not loosen
numerical tolerances just to make a failing check pass.

.. list-table:: Symptoms, checks and recovery
    :header-rows: 1
    :widths: 24 40 36

    * - Symptom
      - Minimal diagnostic
      - Recovery and contract
    * - Notebook behaves differently from a terminal
      - Print ``sys.executable``, ``astropop.__file__`` and ``astropop.__version__``.
      - Select the kernel in the intended environment, restart it, rerun from
        the beginning. See :doc:`../installation`.
    * - Disabling uncertainty did not reduce memory
      - Check whether frames were created before setting the class attribute.
      - Set configuration before loading; reconstruct frames if necessary.
        Existing hidden arrays are not freed. See :doc:`configuration`.
    * - Addition/subtraction raises a units error
      - Inspect both operand units, including bare numbers being dimensionless.
      - Supply a compatible Quantity or calibrated units; do not erase units
        to bypass the check. See :doc:`../how_to/image_arithmetic`.
    * - FITS input has missing/wrong data or uncertainty
      - Inspect HDU names, extension shapes and unit header cards.
      - Select extensions explicitly and verify standard-deviation convention.
        See :doc:`../data_structures/framedata`.
    * - Mask or reason bits disappear
      - Inspect ``flags`` and ``mask`` before and after every operation and
        check whether flag storage is disabled.
      - Calibration clears flags; FITS interchange keeps Boolean masks.
        Adopt an explicit policy using :doc:`../tutorials/quality_propagation`.
    * - Combination pixel is NaN or masked
      - Count accepted samples and inspect input masks, NaNs and clipping limits.
      - With no accepted samples, output is masked; missing-value fill follows
        dtype. Fix rejection/data selection, not the missing-value indicator.
        See :doc:`../how_to/image_combination`.
    * - Combined integer values lack fractional precision
      - Check output dtype and expected range.
      - Temporary calculations may promote, but storage casts to the selected
        type. Choose floating storage when fractional results matter.
    * - Resident memory exceeds ``memory_limit``
      - Count caller-owned and internal images, optional arrays and outputs.
      - The limit covers work tiles, not the whole process. Use paths, smaller
        storage dtype or disk caching. See :ref:`imcombine-memory-planning`.
    * - Central registration succeeds but edge stars drift
      - Compare coordinate residuals at several field radii and inspect bright
        sources entering/leaving the window.
      - A translation cannot remove distortion. See :doc:`../tutorials/registration_diagnostics`
        before interpreting a small correlation shift as full-field accuracy.
    * - Registration raises or has invalid shifts
      - Check overlap, source contrast, masks, window bounds and failure metadata.
      - Exclude failed estimates from products/plots or rerun with an appropriate
        algorithm and well-populated window. See :doc:`../reduction/registering`.
    * - No sources detected
      - Inspect sky/noise, threshold, FWHM, masks and the detection wrapper used.
      - Wrappers do not share an empty-table contract. See
        :ref:`empty-detection-behavior` for current warnings and failures.
    * - Catalog request fails or returns no rows
      - Distinguish service exception from successful zero-length result;
        inspect query region, table/column names and saved response.
      - Use a reviewed local snapshot while diagnosing service health. Do not
        turn a timeout into a successful empty catalog. See :doc:`catalog_queries`.
    * - Astrometry solver cannot start or solve
      - Check the ``solve-field`` executable, index paths, scale/position hints
        and solver output.
      - Install the external program/indexes and choose applicable hints;
        pip does not install them. See :doc:`../reduction/astrometry`.
    * - Grouping FITS files says a key column is missing
      - Inspect ``group.summary.colnames``.
      - Use exact case for ``group_by`` even if header filtering accepted
        uppercase names. See :doc:`../tutorials/fits_collections`.

When reporting a suspected bug, include a tiny reproducible input or synthetic
example, expected versus actual result, version information and full traceback.
State whether external services, local indexes or instrument-specific data are
required. See :doc:`../dev/testing` for reproducing offline versus live checks.
