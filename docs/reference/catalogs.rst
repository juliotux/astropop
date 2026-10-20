.. include:: ../references.txt

Catalogs
========

Use these entries for signatures, parameter choices and return contracts.
For workflows and scientific assumptions, read the linked guides.

Guide: :doc:`../tools/catalogs`.

.. automodapi:: astropop.catalogs
    :no-inheritance-diagram:

Guide: :doc:`../tools/catalog/gaia`.

.. automodapi:: astropop.catalogs.gaia
    :skip: gaiadr3
    :no-inheritance-diagram:
    :inherited-members:

.. currentmodule:: None

.. py:data:: astropop.catalogs.gaia.gaiadr3

    Alias of :class:`~astropop.catalogs.gaia.GaiaDR3SourcesCatalog`; accepts the same
    constructor arguments and returns the same catalog type.

Guide: :doc:`../tools/catalog/simbad`.

.. automodapi:: astropop.catalogs.simbad
    :skip: simbad
    :no-inheritance-diagram:
    :inherited-members:

.. currentmodule:: None

.. py:data:: astropop.catalogs.simbad.simbad

    Alias of :class:`~astropop.catalogs.simbad.SimbadSourcesCatalog`; accepts the same
    constructor arguments and returns the same catalog type.

Guide: :doc:`../tools/catalog/vizier`.

.. automodapi:: astropop.catalogs.vizier
    :skip: list_vizier_catalogs
    :no-inheritance-diagram:
    :inherited-members:

.. currentmodule:: astropop.catalogs.vizier

.. py:function:: list_vizier_catalogs()

    Return a formatted string listing preconfigured catalog names, descriptions,
    and reference bibcodes from the packaged YAML files. Print the returned
    string to inspect the installed choices. No online query is performed.
