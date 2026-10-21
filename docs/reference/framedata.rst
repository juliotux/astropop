.. include:: ../references.txt

FrameData and configuration
===========================

Guide: :doc:`../data_structures/framedata`.

.. automodapi:: astropop.framedata
    :no-inheritance-diagram:

.. automodapi:: astropop.config
    :no-inheritance-diagram:

Guide: :doc:`../data_structures/cache`.

.. automodapi:: astropop.framedata.cache_manager
    :no-inheritance-diagram:

.. currentmodule:: None

.. py:data:: astropop.framedata.cache_manager.BaseTempDir

    Shared TempDir instance at ``CACHE_DIR``. Relative top-level manager names
    are created beneath this instance.

.. currentmodule:: None

.. py:data:: astropop.framedata.cache_manager.CACHE_DIR

    Path beneath Astropy's cache directory used for Astropop-managed files.
    The value is established when the module is imported.

.. currentmodule:: None

.. py:data:: astropop.framedata.cache_manager.DELETE_ON_EXIT

    Module-level Boolean, initially ``True``, controlling shutdown cleanup.
    Individual entries' retention policies still apply when cleanup is enabled.
