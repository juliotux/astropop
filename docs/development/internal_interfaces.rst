Internal extension interfaces
=============================

The bases below explain the implementation hierarchy referenced by the public
API. They are internal extension points, not additional user-facing reduction
functions. Applications should normally use the concrete registration, online
catalog, and polarimetry classes documented in the module chapters.

Registration
------------

.. autoclass:: astropop.image.register._BaseRegister

Online catalogs
---------------

.. autoclass:: astropop.catalogs._sources_catalog._OnlineSourcesCatalog

Dual-beam polarimetry
---------------------

.. autoclass:: astropop.polarimetry._DualBeamPolarimetry
