Shared annotation contracts
============================

Function annotations are the source of parameter and return types. Docstrings
explain behavior, units, shape restrictions, defaults and errors. Shared aliases
keep signatures readable; they do not perform runtime validation or replace the
operation-specific requirements. They live in the private implementation module
``astropop._typing`` and can change with the implementation.

.. py:data:: astropop._typing.ImageInput

    Image-like values accepted by conversion helpers: arrays, frames, quantities,
    CCDData, FITS HDUs/lists or file paths. Individual operations may accept a
    narrower subset; see :doc:`../data_structures/framedata`.

.. py:data:: astropop._typing.QuantityInput

    Values convertible to QFloat, including arrays, frames, quantities and units.
    Conversion does not waive compatible-unit requirements in arithmetic.

.. py:data:: astropop._typing.NumberArray

    A Python or NumPy number, or a NumPy array. Shapes and real/complex dtype
    restrictions depend on the operation.

.. py:data:: astropop._typing.UnitLike

    A unit string, Astropy unit or Quantity, or None. See
    :doc:`../data_structures/physical` for unit conversion.

.. py:data:: astropop._typing.HeaderLike

    An Astropy FITS Header or a string-keyed mapping. FITS serialization
    imposes additional keyword and value restrictions.

.. py:data:: astropop._typing.PathLike

    A string path or an object implementing the string-valued path protocol,
    such as pathlib.Path.

.. py:data:: astropop._typing.ImageHDU

    An Astropy PrimaryHDU, ImageHDU or CompImageHDU. Its data may still be absent;
    see :doc:`../tools/fits_utils` for image-extension selection.

.. py:data:: astropop._typing.Index

    Array indexing syntax: integer, slice, array-like selection, ellipsis or
    a tuple of indices. Each container defines its supported shapes.

.. py:data:: astropop._typing.Axis

    A single integer axis, a tuple of axes or None. The called reduction
    determines whether None means all axes.

.. py:data:: astropop._typing.Window

    A pair of row and column slices, or None for the full image. See
    :doc:`../reduction/registering` for bounds and transform coordinates.

.. py:data:: astropop._typing.Statistic

    A named statistic or a callable accepting arrays. Each caller defines
    the available names and the callable signature.


These are private aliases for interpreting rendered signatures, not imports
required in application code. Consult the linked operation for numerical shape,
unit, copying and missing-value contracts.

.. py:data:: astropop._unit_property._C

    A class type preserved by the unit-property decorator.

.. py:data:: astropop.file_collection._Extension

    A FITS extension number or name, optionally paired with its version.

.. literalinclude:: ../../astropop/file_collection.py
    :language: python
    :start-at: _Extension =
    :end-at: _Extension =
