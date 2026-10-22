Version changes and reproducible reductions
============================================

This guide describes the development checkout, not a promise that every
published release has the same API. Check ``astropop.__version__`` and the
installed metadata before migrating a production reduction. Use the
`GitHub releases <https://github.com/sparc4-dev/astropop/releases>`_ and
`commit history <https://github.com/sparc4-dev/astropop/commits/main/>`_ as the
version-qualified record of changes.

Unreleased changes after 0.9.4
------------------------------

The following contracts are present in this development tree after the
``0.9.4`` tag. In particular, that tag did not yet contain ``astropop.config``;
the optional-array policy was introduced by commit ``47997fb`` and unified
for image reductions by ``c77d1a0``. These are source-history boundaries, not
a new published release number.

* Optional arrays are configured through class attributes on ``AstropopConfig``.
  Set ``FRAMEDATA_DISABLE_UNCERTAINTY`` before constructing/loading frames to omit
  stored errors and skip image uncertainty calculation. There is no separate
  arithmetic-only uncertainty switch. Set ``FRAMEDATA_DISABLE_FLAGS`` separately
  when flag/mask storage is unwanted. See :doc:`../modules/utilities/configuration`.
* Do not rely on masks excluding operands in arithmetic. Combination excludes
  masked samples; calibration convenience functions clear flags; FITS exchange
  retains a Boolean mask rather than every reason bit. See
  :doc:`../tutorials/diagnostics/quality_propagation` before migrating quality-sensitive workflows.
* Mean/median combination uses scatter errors, while sums can propagate input
  errors. Identical nominal inputs can have zero combined scatter despite
  nonzero measurement errors. See :doc:`../modules/framedata/uncertainties`.
* Combination dtype controls stack/output storage while temporary NumPy
  calculations may promote. Code expecting integer accumulators to wrap should
  instead use the documented numerical contract. See
  :doc:`../modules/image/imcombine`.
* This checkout's minimum dependencies are described in :doc:`../getting_started/installation`
  and ``pyproject.toml``. Do not copy their bounds to older releases without
  checking those releases' own metadata. An environment with prerelease
  dependencies is a separate compatibility test, not a supported-version proof.

For example, migrate an application-specific “skip arithmetic errors” setting
to a process policy before any input is read::

    from astropop.config import AstropopConfig as conf
    from astropop.framedata import read_framedata

    conf.FRAMEDATA_DISABLE_UNCERTAINTY = True

    # Set the allocation policy before loading pixels and their optional arrays.
    frame = read_framedata('science.fits')

This static recipe requires your file and deliberately disables all FrameData
image uncertainty calculation. If later stages require errors, use a consistent
enabled policy instead; toggling visibility does not reconstruct lost errors.

Record enough to repeat the result
-----------------------------------

Save the exact package version/commit, Python and dependency versions, detector
unit/gain conventions, optional-array settings, input/calibration identities,
rejection and normalization settings, dtype and memory/cache settings,
registration algorithm/window/reference, photometry apertures/noise model,
coordinate frames/epochs and catalog release/query date. A citation alone does
not identify a reproducible software environment. See :doc:`citation`.
