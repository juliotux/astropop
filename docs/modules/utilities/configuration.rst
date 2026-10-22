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

For example, disable uncertainty storage before constructing a frame and
restore the process setting afterward. The frame created in this scope has no
stored uncertainty, even though its constructor receives one::

    >>> from astropop.config import AstropopConfig
    >>> from astropop.framedata import FrameData

    >>> previous = AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY

    >>> try:
    ...
    ...     # Disable allocation before constructing the frame, even when errors are supplied.
    ...     AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = True
    ...     frame = FrameData([[100., 101.]], uncertainty=2., unit='adu')
    ...     print('Uncertainty:', frame.uncertainty)
    ... finally:
    ...     AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = previous
    Uncertainty: None

See :ref:`imcombine-memory-planning` for storage estimates and
:doc:`../framedata/index` for getters, setters and copy behavior.


Input and result diagnostics
----------------------------

.. list-table:: Symptoms, checks and recovery
    :header-rows: 1
    :widths: 24 40 36

    * - Symptom
      - Minimal diagnostic
      - Recovery and contract
    * - Disabling uncertainty did not reduce memory
      - Check whether frames were created before setting the class attribute.
      - Set configuration before loading; reconstruct frames if necessary.
        Existing hidden arrays are not freed. See :doc:`configuration`.

Reference/API
-------------

.. automodapi:: astropop.config
    :no-inheritance-diagram:


.. currentmodule:: None
