.. include:: ../../references.txt

Building a reduction workflow
=============================

Astropop supplies reduction functions; your application owns observation
selection, detector policy, processing order, failures and output provenance.
Use :doc:`index` for the numerical CCD sequence. This guide
focuses on organizing those operations into a repeatable application.

.. _a-small-calibration-workflow:

Select compatible inputs
------------------------

Build an observing log with :class:`~astropop.file_collection.FitsFileGroup`.
Use your instrument's header names to separate camera/channel, binning, readout
mode, amplifier layout, crop/shape and detector temperature. Flats also need the
appropriate filter/optical configuration. A matching array shape alone does not
establish calibration compatibility.

Inspect the selected counts and exposure times before building masters. Bias
frames define an additive zero; bias-subtracted darks need a stated exposure
scale; flats describe a response after their own additive corrections. See
:doc:`../../modules/image/processing` for the operations. The tutorials use one detector setup and
select filename groups after inspection; an application should use its reviewed
log rather than assuming every file named ``flat*`` belongs in one master.

Store explicit input/output lists per stage. A rerun with a changed crop,
registration reference, calibration selection or measurement radius invalidates
downstream products even if their filenames already exist.

Order, registration, and quality
--------------------------------

Apply compatible geometry to science and calibration frames. Correct detector
effects before measuring stars or estimating registration transforms. Register
shifted science images before combining them. Photometry consumes the resulting
image coordinate system; polarimetry additionally needs beam identification and
retarder angles. Overscan and nonlinearity corrections are detector-specific
steps that the application must provide when needed.

Choose quality propagation explicitly. Arithmetic marks pixels without excluding
their values, combination rejects masked samples, and calibration convenience
functions currently clear flag bits. Preserve and combine science/master quality
maps according to :doc:`../diagnostics/quality_propagation` rather than inferring usable
pixels from a successful numerical operation.

Failures and restart policy
---------------------------

Treat a failed frame, an empty measurement and a failed catalog request as
separate outcomes. Record the filename, stage, exception and processing settings.
Catch only errors for which the application has an explicit recovery policy;
see :doc:`../../getting_started/troubleshooting`. Do not replace failed science results with
empty successful products.

Write derived products to a separate output directory and choose overwrite
behavior deliberately. For long runs, a stage manifest can record completed
outputs and their input identities. Before resuming, verify those inputs and
settings still match; file existence alone is not a freshness check. If atomic
publication is required, write and validate a temporary file on the destination
filesystem before renaming it to the final path.

Large datasets and reproducibility
----------------------------------

Configure optional arrays before constructing frames. Workers in separate
processes need their own configuration before loading data; avoid threads that
mutate shared configuration to different policies. See
:doc:`../../modules/utilities/configuration`.

Keep calibration masters for reuse, but iterate over science files and release
intermediate images when no longer needed. Memmaps and the combiner's disk cache
move storage to disk. The combination memory budget covers work tiles, not all
caller-owned frames or output arrays; see :ref:`imcombine-memory-planning`.

Record software versions, input identities, configuration, calibration filenames,
exposure/gain choices, rejection settings and registration reference with the
products. For photometry, also record aperture, annulus, background and flag
selection choices; for catalogs, retain release, epoch and query provenance.
The numerical arrays alone cannot reconstruct these decisions.
