Shared annotation contracts
============================

Function annotations are the source of parameter and return types. Docstrings
explain behavior, units, shape restrictions, defaults and errors. Shared aliases
keep signatures readable; they do not perform runtime validation or replace the
operation-specific requirements. They live in the private implementation module
``astropop._typing`` and can change with the implementation.

Reference/API
-------------

.. py:data:: astropop._typing.ImageLike

    A two-dimensional pixel array with shape ``(rows, columns)``, including
    nested numerical sequences and NumPy arrays. This is an ``Annotated``
    NumPy ``ArrayLike`` contract with semantic shape metadata; type checkers do
    not enforce the number of dimensions. Each operation's documented shape
    requirements still apply. It does not include filenames or FrameData
    containers, and it does not prescribe a dtype, copy or runtime conversion.
    Masks, flag planes and uncertainty maps use this same spatial convention;
    scalar uncertainties are listed separately where supported.

.. py:data:: astropop._typing.ImageInput

    Image-like values accepted by conversion helpers: arrays, frames, quantities,
    CCDData, FITS HDUs/lists or file paths. Individual operations may accept a
    narrower subset; see :doc:`../framedata/index`.

.. py:data:: astropop._typing.QuantityInput

    Values convertible to QFloat, including arrays, frames, quantities and units.
    Conversion does not waive compatible-unit requirements in arithmetic.

.. py:data:: astropop._typing.NumberArray

    A Python or NumPy number, or a NumPy array. Shapes and real/complex dtype
    restrictions depend on the operation.

.. py:data:: astropop._typing.UnitLike

    A unit string, Astropy unit or Quantity, or None. See
    :doc:`../quantities/index` for unit conversion.

.. py:data:: astropop._typing.HeaderLike

    An Astropy FITS Header or a string-keyed mapping. FITS serialization
    imposes additional keyword and value restrictions.

.. py:data:: astropop._typing.PathLike

    A string path or an object implementing the string-valued path protocol,
    such as pathlib.Path.

.. py:data:: astropop._typing.ImageHDU

    An Astropy PrimaryHDU, ImageHDU or CompImageHDU. Its data may still be absent;
    see :doc:`../files/fits_utils` for image-extension selection.

.. py:data:: astropop._typing.Index

    Array indexing syntax: integer, slice, array-like selection, ellipsis or
    a tuple of indices. Each container defines its supported shapes.

.. py:data:: astropop._typing.Axis

    A single integer axis, a tuple of axes or None. The called reduction
    determines whether None means all axes.

.. py:data:: astropop._typing.Window

    A pair of row and column slices, or None for the full image. See
    :doc:`../image/register` for bounds and transform coordinates.

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

.. literalinclude:: ../../../astropop/file_collection.py
    :language: python
    :start-at: _Extension =
    :end-at: _Extension =
