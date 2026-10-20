Shared annotation contracts
============================

Function annotations are the source of parameter and return types. Docstrings
explain behavior, units, shape restrictions, defaults and errors. Shared aliases
keep signatures readable; they do not perform runtime validation or replace the
operation-specific requirements. They live in the private implementation module
``astropop._typing`` and can change with the implementation.

.. py:data:: astropop._typing.ImageInput

.. py:data:: astropop._typing.QuantityInput

.. py:data:: astropop._typing.NumberArray

.. py:data:: astropop._typing.UnitLike

.. py:data:: astropop._typing.HeaderLike

.. py:data:: astropop._typing.PathLike

.. py:data:: astropop._typing.ImageHDU

.. py:data:: astropop._typing.Index

.. py:data:: astropop._typing.Axis

.. py:data:: astropop._typing.Window

.. py:data:: astropop._typing.Statistic


The definitions below are included from the implementation, so the accepted
unions have one source of truth. Imports with a leading underscore are local
names for NumPy, Astropy and standard-library types.

.. literalinclude:: ../../astropop/_typing.py
    :language: python
    :start-at: from os import PathLike
    :end-at: Statistic: _TypeAlias

.. py:data:: astropop._unit_property._C

    A class type preserved by the unit-property decorator.

.. py:data:: astropop.file_collection._Extension

    A FITS extension number or name, optionally paired with its version.

.. literalinclude:: ../../astropop/file_collection.py
    :language: python
    :start-at: _Extension =
    :end-at: _Extension =
