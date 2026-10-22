.. include:: ../references.txt

Quantities, models, files and utilities
=======================================

Physical quantities
-------------------

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

.. _qfloat-numpy-support:

QFloat NumPy support
^^^^^^^^^^^^^^^^^^^^

These names have QFloat handlers. Support does not imply every NumPy keyword
or ufunc method is available; the guide describes argument limitations and
reduction uncertainty estimators. Unsupported operations should be evaluated
explicitly on suitable numerical inputs with their unit/error policy stated.

Array functions
~~~~~~~~~~~~~~~

- `~numpy.append`
- `~numpy.around`
- `~numpy.clip`
- `~numpy.delete`
- `~numpy.expand_dims`
- `~numpy.flip`
- `~numpy.fliplr`
- `~numpy.flipud`
- `~numpy.insert`
- `~numpy.moveaxis`
- `~numpy.ravel`
- `~numpy.repeat`
- `~numpy.reshape`
- `~numpy.resize`
- `~numpy.roll`
- `~numpy.rollaxis`
- `~numpy.round`
- `~numpy.rot90`
- `~numpy.shape`
- `~numpy.size`
- `~numpy.squeeze`
- `~numpy.swapaxes`
- `~numpy.take`
- `~numpy.tile`
- `~numpy.transpose`

Statistical functions
~~~~~~~~~~~~~~~~~~~~~

``sum``, ``mean``, ``median``, ``nanmean``, ``nanmedian``, ``std``, ``nanstd``,
``var`` and ``nanvar`` are supported. See
:doc:`../data_structures/physical` for the different uncertainty contracts.
``nansum`` is not implemented.

Universal functions
~~~~~~~~~~~~~~~~~~~

- `~numpy.absolute`
- `~numpy.add`
- `~numpy.arccos`
- `~numpy.arccosh`
- `~numpy.arcsin`
- `~numpy.arcsinh`
- `~numpy.arctan`
- `~numpy.arctan2`
- `~numpy.arctanh`
- `~numpy.ceil`
- `~numpy.copysign`
- `~numpy.cos`
- `~numpy.cosh`
- `~numpy.deg2rad`
- `~numpy.degrees`
- `~numpy.divide`
- `~numpy.divmod`
- `~numpy.exp`
- `~numpy.exp2`
- `~numpy.expm1`
- `~numpy.float_power`
- `~numpy.floor`
- `~numpy.floor_divide`
- `~numpy.hypot`
- `~numpy.isfinite`
- `~numpy.isinf`
- `~numpy.isnan`
- `~numpy.log`
- `~numpy.log10`
- `~numpy.log1p`
- `~numpy.log2`
- `~numpy.multiply`
- `~numpy.negative`
- `~numpy.positive`
- `~numpy.power`
- `~numpy.rad2deg`
- `~numpy.radians`
- `~numpy.remainder`
- `~numpy.sign`
- `~numpy.signbit`
- `~numpy.sin`
- `~numpy.sinh`
- `~numpy.sqrt`
- `~numpy.square`
- `~numpy.subtract`
- `~numpy.tan`
- `~numpy.tanh`
- `~numpy.trunc`

``abs`` aliases ``absolute``; ``mod`` aliases ``remainder``; ``true_divide``
aliases ``divide``. Predicate/sign ufuncs such as ``isfinite`` and ``sign``
inspect nominal values and return ordinary NumPy results. They do not test the
uncertainty array. ``floor``, ``ceil`` and ``trunc`` return QFloat and round
its uncertainty to zero decimal places. ``arctan2(y, x)`` accepts compatible
units and returns radians, unlike the dimensionless-input inverse functions.

File collections
----------------

Guide: :doc:`../tools/files`.

.. automodapi:: astropop.file_collection
    :no-inheritance-diagram:

Flags and metadata merging
--------------------------

Guide: :doc:`../tools/flags`.

.. automodapi:: astropop.flags
    :no-inheritance-diagram:

.. automodapi:: astropop.image._tools
    :no-inheritance-diagram:

Array helpers and analytical models
-----------------------------------

Guide: :doc:`../tools/math`.

.. automodapi:: astropop.math.array
    :no-inheritance-diagram:

.. automodapi:: astropop.math.models
    :no-inheritance-diagram:

.. automodapi:: astropop.math.hasher
    :no-inheritance-diagram:

FITS utilities
--------------

Guide: :doc:`../tools/fits_utils`.

.. automodapi:: astropop.fits_utils
    :no-inheritance-diagram:

.. autofunction:: astropop.fits_utils.string_to_header_key

.. currentmodule:: None

.. py:data:: astropop.fits_utils.imhdus

    Tuple of Astropy image-HDU classes suitable for ``isinstance`` checks,
    as illustrated in :doc:`../tools/fits_utils`. An HDU can belong to one of these classes and still
    contain no image data.

Logging
-------

Guide: :doc:`../tools/logger`.

.. automodapi:: astropop.logger
    :no-inheritance-diagram:

.. currentmodule:: None

.. py:data:: astropop.logger.logger

    Shared standard-library Logger named ``astropop``. Its initial level is
    ``INFO``; changing its level affects subsequent library log messages.

Processes and Python helpers
----------------------------

Guide: :doc:`../tools/py_utils`.

.. automodapi:: astropop.py_utils
    :no-inheritance-diagram:

Test helpers
------------

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
