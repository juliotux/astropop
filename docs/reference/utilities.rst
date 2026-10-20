.. include:: ../references.txt

Quantities, models, files and utilities
=======================================

Use these entries for signatures, parameter choices and return contracts.
For workflows and scientific assumptions, read the linked guides.

Guide: :doc:`../data_structures/physical`.

.. automodapi:: astropop.math.physical
    :skip: UnitsError
    :no-inheritance-diagram:

.. currentmodule:: None

.. py:data:: astropop.math.physical.units

    Alias of :mod:`astropy.units`, available for constructing units and
    quantities used by QFloat. This is not a separate unit registry.

.. currentmodule:: None

.. py:exception:: astropop.math.physical.UnitsError

    Alias of :class:`astropy.units.UnitsError`, used for incompatible physical
    units. It is the Astropy exception class, not a distinct Astropop subtype.

Guide: :doc:`../tools/files`.

.. automodapi:: astropop.file_collection
    :no-inheritance-diagram:

Guide: :doc:`../tools/flags`.

.. automodapi:: astropop.flags
    :no-inheritance-diagram:

.. automodapi:: astropop.image._tools
    :no-inheritance-diagram:

Guide: :doc:`../tools/math`.

.. automodapi:: astropop.math.array
    :no-inheritance-diagram:

.. automodapi:: astropop.math.models
    :no-inheritance-diagram:

.. automodapi:: astropop.math.hasher
    :no-inheritance-diagram:

Guide: :doc:`../tools/fits_utils`.

.. automodapi:: astropop.fits_utils
    :no-inheritance-diagram:

.. autofunction:: astropop.fits_utils.string_to_header_key

.. currentmodule:: None

.. py:data:: astropop.fits_utils.imhdus

    Tuple of Astropy image-HDU classes suitable for ``isinstance`` checks,
    as illustrated above. An HDU can belong to one of these classes and still
    contain no image data.

Guide: :doc:`../tools/logger`.

.. automodapi:: astropop.logger
    :no-inheritance-diagram:

.. currentmodule:: None

.. py:data:: astropop.logger.logger

    Shared standard-library Logger named ``astropop``. Its initial level is
    ``INFO``; changing its level affects subsequent library log messages.

Guide: :doc:`../tools/py_utils`.

.. automodapi:: astropop.py_utils
    :no-inheritance-diagram:

Guide: :doc:`../tools/testing`.

.. automodapi:: astropop.testing
    :skip: assert_warns, assert_not_warnings
    :no-inheritance-diagram:

.. currentmodule:: astropop.testing

.. py:function:: assert_warns(warning_class, *args, **kwargs)

    Alias of :func:`numpy.testing.assert_warns`. Use it as a context manager to
    require a warning, or pass a callable and its arguments to invoke it under
    the warning check. The assertion fails if the requested warning does not
    occur.

.. py:function:: assert_not_warnings(*args, **kwargs)

    Alias of :func:`numpy.testing.assert_no_warnings`. Use it as a context
    manager or pass a callable and arguments to require warning-free execution.
