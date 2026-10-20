Configuration scope and allocation
===================================

Set attributes on ``AstropopConfig`` itself; do not instantiate it. Apply the
settings **before loading or creating frames** so optional arrays are not
allocated. Disabling arrays later hides existing arrays without freeing them.
Re-enabling reveals the old arrays, which may no longer describe data modified
while they were hidden. Prefer a consistent policy for an entire reduction.

``FRAMEDATA_DISABLE_UNCERTAINTY=True`` omits uncertainty extraction, new storage,
and image-reduction uncertainty calculation. ``FRAMEDATA_DISABLE_FLAGS=True``
independently omits flag/mask storage. With masks disabled, masked-pixel exclusion
cannot rely on FrameData masks. Neither option changes physical unit checking.

For temporary use, save and restore the class attributes in ``try/finally`` as
shown in :ref:`optional-frame-arrays`. This restores the process settings, not
arrays that were never stored. Avoid concurrent threads using different policies:
these are not task-local or thread-local settings. A spawned worker imports a
fresh interpreter state; configure each worker before it loads its frames.
Pass the policy explicitly in a worker initializer rather than relying on the
parent's class attributes being inherited.

See :doc:`../how_to/large_datasets` for storage estimates and
:doc:`../data_structures/framedata` for getters, setters and copy behavior.
