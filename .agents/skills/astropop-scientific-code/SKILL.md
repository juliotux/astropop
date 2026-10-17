---
name: astropop-scientific-code
description: Implement and test Astropop scientific behavior in FrameData, image reduction, units and uncertainties, photometry, or polarimetry while preserving numerical and storage contracts.
---

# Astropop scientific code

Locate the affected code in `astropop/framedata/`, `astropop/image/`, `astropop/math/`, `astropop/photometry/`, or `astropop/polarimetry/` and its existing tests. Read the current API and documentation; do not assume optional uncertainties, cache redesigns, or registration windows exist because they were discussed on another branch.

## Define observable behavior

Identify the expected data, units, shape/dtype, mask, uncertainty, metadata, and input-mutation behavior. Add a small numerical regression case with an independently derived result. Include compatible unit conversion and incompatible-unit errors when relevant.

- Missing uncertainty and zero uncertainty are different storage states. If an operation promises to skip uncertainty work, test that conversion/propagation helpers are not invoked, not merely that the final array is absent.
- In combining, apply rejection consistently to data and propagated errors. Check valid-sample counts and any normalization factor, including masked/all-rejected pixels. Preserve the existing estimator unless a statistical change is requested.
- Test in-place and copying paths when affected; non-inplace calls must preserve source arrays and metadata. Exercise RAM/memmap paths for storage changes, including cleanup after exceptions.
- Registration uses array row/column conventions and transform coordinate conventions that may differ. For a requested window feature, estimate from the window but test application to the full image and any necessary origin translation. Use sparse, deterministic synthetic sources to avoid ambiguous matches.
- Photometry fixtures should use seeded noise when noise is necessary. Verify scalar and array-valued parameters against each other when both are supported. Do not simply loosen a numerical tolerance to hide nondeterminism.

Use the existing test module for the affected subsystem. Restore global configuration changes with fixtures/monkeypatch. Keep regression cases small enough to diagnose and run routinely; use broader integration tests when the change crosses subsystem boundaries.

Document changed scientific assumptions and API behavior, and distinguish numerical correctness checks from performance measurements. Do not claim speed or memory improvements without measurements when those claims matter to the task.
