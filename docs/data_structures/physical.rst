.. include:: ../references.txt

.. |QFloat.uncertainty| replace:: `~astropop.math.physical.QFloat.uncertainty`
.. |QFloat.nominal| replace:: `~astropop.math.physical.QFloat.nominal`
.. |QFloat.unit| replace:: `~astropop.math.physical.QFloat.unit`

Physical Quantities and Uncertainties
=====================================

Any physical measurement is not a simple number. It is a number, but have a physical unit and an uncertainty related to it.

|astropy| has a very useful module called `~astropy.units`, that handles physical units and |Quantity|, a container intended to store physical measurements with units.. But it doesn't perform any error propagation.

To handle this and make the things a lot easier in |astropop|, we created a new class, called |QFloat| to handle both uncertainties and physical units at once. By |QFloat| we mean "Quantity Float", relating it to physical measurements. This class mainly wraps `~astropy.units` methods to handle units and perform error propagation in a coherent way. This module is used in a lot of places inside |astropop|, mainly in `~astropop.image.processing` and |Framedata| to ensure correct processing in terms of units and errors propagation.

.. WARNING:: In the actual state, |QFloat| assumes all uncertainties are standard deviation errors, performing the standard error propagation method. The error propagation also assumes that the uncertainties are uncorelated. This is fine for our usage, but may present problems if used out of this context.

The |QFloat| Class
------------------

The |QFloat| class stores basically 3 variables: the nominal value, the uncertainty and the physical unit. To create this class, just pass these values in the class constructor and you can access it using the proper properties.

.. ipython:: python

    from astropop.math import QFloat
    import numpy as np
    # A physical measure of 1.000+/-0.001 meters
    qf = QFloat(1.0, 0.001, 'm')
    qf.nominal # the nominal value, must be 1.0
    qf.uncertainty # the uncertainty, 0.001
    qf.unit # astropy's physical unit, meter
    qf # full representation.

Note that, for the full representation of the quantity, the nominal and the uncertainty values are rounded to the first non-zero error decimal. Internally, the number is stored with all the computed decimal places and this rounding just appears in the string representation.

.. ipython:: python

    qf = QFloat(1.0583225, 0.0031495756, 'cm')
    qf.nominal
    qf.uncertainty
    qf

|QFloat| also can store arrays of data, using the exactly same behavior.

.. ipython:: python

    qf = QFloat([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], 'm/s')
    qf

During the creation, you can omit |QFloat.uncertainty| or |QFloat.unit| arguments, but not the nominal value. We decided to don't make possible create empty |QFloat| instances. Omiting arguments, the code interprets it as:

- |QFloat.unit|: setting `None` unit, or omiting it, the class automatically interpret it as |dimensionless_unscaled|. This means: a number without physical unit.

.. ipython:: python

    qf_nounit = QFloat(1.0, 0.1)
    qf_nounit

- |QFloat.uncertainty|: setting uncertainties as `None`, or omiting it, the code consider automatically the uncertainty as 0.0.

.. ipython:: python
    :okwarning:

    qf_nostd = QFloat(1.0, unit='m')
    qf_nostd

You also can omit both, like converting a single dimensionless number to QFloat.

.. ipython:: python
    :okwarning:

    qf = QFloat(1.0)
    qf

Printing and String Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the |QFloat| shows the numbers rounded to the first significant digit of the uncertainty (rounded). This is done to make the representation more readable. But you can set the number of significant digits to show by setting the `~astropop.math.physical.QFloat.sig_digits` property.

.. ipython:: python

    qf = QFloat(1.0583225, 0.0031495756, 'cm')
    print(qf)
    qf.sig_digits = 2
    print(qf)
    qf.sig_digits = 3
    print(qf)

Units and Conversion
--------------------

Physical units are fully handled with `~astropy.units` module. But we don't use |Quantity| class to store the files. Instead, the code perform itself the units checking and conversions. This is needed to make it compatible with uncertainties, but reduces the compatibility with |astropy| functions directly.

Internal conversion of units for math operations are made automatically and don't need manual intervention. But manual conversions are also possible using the ``<<`` operator or `~astropop.math.physical.QFloat.to` function. The same do the same thing: convert the actual |QFloat| to a new one, with the ne unit. Both accept `~astropy.units.UnitBase` instance or string for conversion.

.. ipython:: python

    from astropop.math import QFloat
    from astropy import units
    # One kilometer with 1 cm uncertainty
    qf = QFloat(1000, 0.01, 'm')
    qf << 'km'
    qf << units.cm

If improper conversion (incompatible units), an `~astropy.units.UnitConversionError` is raised.

Math Operations and Error Propagation
-------------------------------------

As the main porpouse of this module, mathematical operations using physical quantities are performed between |QFloat| and compatible classes. We ensure a basic set of math operations to work, specially all the basic and trigonometric operations needed for basic data reduction. The code also support some basic |numpy| array functions, but not all of them.

The operations are performed with proper unit management and conversion (when necessary), and simplyfied uncorrelated error propagation. For a function :math:`f` of :math:`x, y, z, \cdots` variables, the error :math:`\sigma` associated to each one of them, is propagated using the common equation:

.. math::
    \sigma_f = \sqrt{ \left(\frac{\partial f}{\partial x}\right)^2 \sigma_x^2 + \left(\frac{\partial f}{\partial y} \right)^2 \sigma_y^2 + \left(\frac{\partial f}{\partial z} \right)^2 \sigma_z^2 + \cdots}

Astropop's derivative helpers implement this independent-input model. Covariance
is not tracked, including when the same QFloat is used in both operands. The
implementation is inspired by the ``uncertainties`` package but does not provide
that package's correlation tracking. Statistical reductions use the separate
scatter estimators described below.

Supported Math Operations
^^^^^^^^^^^^^^^^^^^^^^^^^

Since the math operations are the main reason of the |QFloat| to exist, they have a special focus in the implementation. All builtin Python math operations, with the exception of matrix multiplication, is implemented for |QFloat|. This makes possible to perform direct math operations with |QFloat|.

As example, to sum two |QFloat|, you just need to use the ``+`` operator.

.. ipython:: python

    qf1 = QFloat(1.0, 0.1, 'm')
    qf2 = QFloat(2.0, 0.1, 'm')
    qf1 + qf2

|astropop| handles all the needed units checking and conversions and the result is dimensionality correct. For example:

.. ipython:: python

    qf1 = QFloat(60, 0.5, 'km')
    qf2 = QFloat(7000, 700, 'm')
    qf1 + qf2
    t = QFloat(2.0, 0.1, 'h')
    qf1/t

Incorrect dimensionality in operations will raise |UnitsError|.

.. ipython::
    :verbatim:

    In [2]: qf1 = QFloat(3.0, 0.01, 'kg')

    In [3]: qf2 = QFloat(5.0, 0.2, 'K')

    In [4]: qf1 + qf2
    UnitConversionError: Can only apply 'decorator' function to quantities with compatible dimensions

Supported Numpy Array Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some `Numpy array functions <https://numpy.org/doc/1.19/reference/routines.array-manipulation.html>`_ are supported built-in by the |QFloat|, but not all of them. These functions are intended to perform array manipulations, like append, reshape, transpose, etc. With this compatibility you can use the Numpy functions directly with |QFloat| objects, like:

.. ipython:: python

    qf = QFloat(np.zeros((100, 100)), np.zeros((100, 100)))
    np.shape(qf)

Array transformations returning a |QFloat| construct new nominal and uncertainty
arrays. Do not assume NumPy's view-sharing behavior: both arrays contribute to
allocation costs. Shape queries return ordinary shape information. Measure
runtime and peak memory for your own array sizes before choosing a workflow.

The current Numpy array functions supported for this class are:

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

Statistical reductions
^^^^^^^^^^^^^^^^^^^^^^

The following reductions accept ``axis=None`` (all elements), an integer axis,
or a tuple of axes. Their handlers accept only ``axis`` beyond the input;
``dtype``, ``out``, ``keepdims``, ``where``, and ``ddof`` are not supported and
raise ``TypeError``. Population standard deviation is used (``ddof=0``).

.. list-table:: Reduction contracts
    :header-rows: 1
    :widths: 22 25 53

    * - Function
      - Output
      - Uncertainty or interpretation
    * - ``sum``
      - QFloat in the input unit
      - Quadrature sum of input standard deviations.
    * - ``mean``
      - QFloat in the input unit
      - Population scatter of nominal values divided by the square root of
        the number of elements; input uncertainties are ignored.
    * - ``median``
      - QFloat in the input unit
      - NaN-ignoring population scatter divided by the square root of the
        total number of elements. Input uncertainties are ignored; the
        nominal median itself does not ignore NaNs.
    * - ``nanmean``
      - QFloat in the input unit
      - NaN-ignoring scatter divided by
        ``sqrt(nansum(nominal) / nanmean(nominal))``.
    * - ``nanmedian``
      - QFloat in the input unit
      - NaN-ignoring scatter divided by
        ``sqrt(nansum(nominal) / nanmedian(nominal))``.
    * - ``std``, ``nanstd``
      - Plain NumPy scalar or array
      - Population scatter of nominal values; units and input uncertainties
        are discarded. Interpret the numbers in the input unit.
    * - ``var``, ``nanvar``
      - Plain NumPy scalar or array
      - Population variance of nominal values; units and input uncertainties
        are discarded. Interpret the numbers in the squared input unit.

The ``nanmedian`` denominator is not generally the number of finite samples.
Both NaN-aware mean/median denominators can be invalid for zero or signed data;
all-NaN slices also produce invalid results and NumPy warnings. These estimators
are not substitutes for propagating measurement errors. ``nansum`` is not
implemented for QFloat.

These local examples check the formulas independently::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropop.math import QFloat
    >>> q = QFloat([[1, 3], [5, 7]], [[2, 2], [2, 2]], 'adu')
    >>> total = np.sum(q, axis=0)
    >>> np.testing.assert_allclose(total.nominal, [6, 10])
    >>> np.testing.assert_allclose(total.uncertainty, np.sqrt([8, 8]))
    >>> assert total.unit == u.adu
    >>> for reduce in (np.mean, np.median):
    ...     result = reduce(q, axis=1)
    ...     np.testing.assert_allclose(result.nominal, [2, 6])
    ...     np.testing.assert_allclose(result.uncertainty, 1 / np.sqrt(2))
    ...     assert result.unit == u.adu
    >>> missing = QFloat([1, 3, 8, np.nan], [100, 100, 100, 100], 'adu')
    >>> scatter = np.sqrt(((1 - 4)**2 + (3 - 4)**2 + (8 - 4)**2) / 3)
    >>> average = np.nanmean(missing)
    >>> np.testing.assert_allclose(average.nominal, 4)
    >>> np.testing.assert_allclose(average.uncertainty, scatter / np.sqrt(3))
    >>> middle = np.nanmedian(missing)
    >>> np.testing.assert_allclose(middle.nominal, 3)
    >>> np.testing.assert_allclose(middle.uncertainty, scatter / np.sqrt(12 / 3))
    >>> assert average.unit == middle.unit == u.adu
    >>> for reduce in (np.std, np.nanstd):
    ...     values = reduce(q, axis=0)
    ...     assert isinstance(values, np.ndarray)
    ...     np.testing.assert_allclose(values, [2, 2])
    >>> for reduce in (np.var, np.nanvar):
    ...     np.testing.assert_allclose(reduce(q, axis=0), [4, 4])
    >>> import pytest
    >>> with pytest.raises(TypeError):
    ...     np.mean(q, keepdims=True)
    >>> with pytest.raises(TypeError):
    ...     np.std(q, ddof=1)

Supported Numpy Universal Functions (UFuncs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QFloat supports the calls below through NumPy's ufunc interface. Only direct
calls are implemented; methods such as ``np.add.reduce`` are unsupported.
``out`` raises ``NotImplementedError``. Other keywords are forwarded to the
individual handler, so support depends on that handler. Arithmetic handlers
such as ``add`` accept no keywords: ``where`` and ``dtype`` raise ``TypeError``.
Do not assume that all NumPy ufunc options work or are silently ignored::

    >>> q = QFloat([1, 3], [0.1, 0.2], 'm')
    >>> with pytest.raises(NotImplementedError):
    ...     np.add(q, q, out=(q,))
    >>> with pytest.raises(TypeError):
    ...     np.add(q, q, where=[True, False])
    >>> with pytest.raises(TypeError):
    ...     np.add.reduce(q)

Use ``np.sum(q)`` for the supported quadrature-error summation.

The current supported ufuncs are:

- `~numpy.absolute` (``numpy.abs``)
- `~numpy.add`
- `~numpy.arccos`
- `~numpy.arccosh`
- `~numpy.arcsin`
- `~numpy.arcsinh`
- `~numpy.arctan`
- `~numpy.arctanh`
- `~numpy.cos`
- `~numpy.cosh`
- `~numpy.deg2rad`
- `~numpy.degrees`
- `~numpy.divide` (`~numpy.true_divide`)
- `~numpy.exp`
- `~numpy.exp2`
- `~numpy.expm1`
- `~numpy.floor_divide`
- `~numpy.hypot`
- `~numpy.log`
- `~numpy.log10`
- `~numpy.log1p`
- `~numpy.log2`
- `~numpy.negative`
- `~numpy.positive`
- `~numpy.power` (`~numpy.float_power`)
- `~numpy.rad2deg`
- `~numpy.radians`
- `~numpy.remainder` (`~numpy.mod`)
- `~numpy.sin`
- `~numpy.sinh`
- `~numpy.square`
- `~numpy.sqrt`
- `~numpy.subtract`
- `~numpy.tan`
- `~numpy.tanh`

All the units checking is automatically performed by |astropop|. In fact, most of these functions just wrap standart operations of |QFloat|, since this class perform the operations in a way very similar to the |ufunc|.

Trigonometric Math
^^^^^^^^^^^^^^^^^^

For trigonometric functions, like sines, cosines and tangents, the code is able to check the dimensionality of the numbers before perform the operation. So, only dimensionless numbers are accepted, being |dimensionless_unscaled| or |dimensionless_angles|. Any number with pyshical dimension don't make any sense inside trigonometric operations, which will raise an |UnitsError|.

To avoid an additional module containing trigonometric functions inside |astropop|, these operations are performed using the |ufunc| described earlier. The main difference here is the unit checking performed by |astropop|.

.. ipython:: python

    qf = QFloat(30, 0.1, 'deg')
    np.cos(qf)
    np.sin(qf)

For trionometric (and hyperbolic) functions, like `~numpy.sin` and `~numpy.sinh`, only angle are accepted. So, only |QFloat| with ``astropy.units.deg`` or ``astropy.units.rad`` will not raise |UnitsError|. Also, all these functions will result in |dimensionless_unscaled| values.

For inverse trigonometric functions, like `~numpy.arcsin`, the inverse happens. The input must be a |dimensionless_unscaled| |QFloat|, and output will be in units of ``astropy.units.rad``.

.. ipython:: python

    qf = QFloat(0.5, 0.01)
    np.arcsin(qf)

Comparisons Notes
-----------------

Comparing two numbers with units and uncertainties is an ambiguous thing. There are multiple ways to consider two numbers equal or different, or even greater or smaller. Due to this, we had to assume some conventions in the processing.

Equality
^^^^^^^^

We consider two numbers equal if they have the same nominal and standard deviation values in the same unit. This means, they are exactly equal in everything, meaning a more programing-like approach. Like:

.. ipython:: python

    # 1.0+/-0.1 meters
    qf1 = QFloat(1.0, 0.1, 'm')
    # same as above, but in cm
    qf2 = QFloat(100, 10, 'cm')
    qf1 == qf2

So, the simple fact that two numers have different error bars imply that they are different.

.. ipython:: python

    # 1.0+/-0.2 meters. Same number, with different error
    qf3 = QFloat(1.0, 0.2, 'm')
    qf1 == qf3
    # 0.5+/-0.1 meters
    qf4 = QFloat(0.5, 0.1, 'm')
    qf1 == qf4

Of course, the different operator works in the exactly same way.

.. ipython:: python

    qf1 != qf2
    qf1 != qf3
    qf1 != qf4

When comparing numbers with same dimension units, the code automatically converts if to compare. But, if incompatible units (different dimensions) are compared, they automatically are considered different. In physical terms, 1 second is different from 1 meter.

.. ipython:: python
    :okwarning:

    # Same nominal values of qf1, but in seconds
    qf5 = (1.0, 0.1, 's')
    qf1 == qf5
    qf1 != qf5

Equality considering errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To check physical equality, which consider if the numbers are equal inside the error bars, we created the `~astropop.math.physical.equal_within_errors` method. In this method we assume two numbers (:math:`a` and :math:`b`) are equal if their different is smaller than the sum of the errors (:math:`\sigma_a` and :math:`\sigma_b`).

.. math::
    | a - b | <= \sigma_a + \sigma_b

In other words, these two numbers are equal if they intercept each other, considering error bars. Or, within the errors, they have ate least one value in common.

So, for a proper physical check of equalities, use `~astropop.math.physical.equal_within_errors` instead of ``==`` operator. For example:

.. ipython:: python

    from astropop.math.physical import QFloat, equal_within_errors
    qf1 = QFloat(1.1, 0.1, 'm')
    qf2 = QFloat(1.15, 0.05, 'm')
    equal_within_errors(qf1, qf2)

Inequalities
^^^^^^^^^^^^

Inequality handling is more ambiguous then equality to define. To avoid a complex API and keep the things in a coherent way, we perform greater, greater or equal, smaller and smaller or equal operations just comparing the nominal values of the numbers. For example:

.. ipython:: python

    qf1 = QFloat(1.1, 0.1, 'm')
    qf2 = QFloat(1.15, 0.05, 'm')
    qf1 < qf2
    qf2 >= qf1
    qf2 < qf1

Note that errors are note being considered in operations. However, this operation perfmors full handling of physical units. So:

.. ipython:: python

    qf1 = QFloat(1.0, 0.1, 'm')
    qf2 = QFloat(50, 10, 'cm')
    qf1 >= qf2

These comparisons can only be performed by same dimension measurements. If incompatible units are used, |UnitsError| is raised.

Physical Quantities API
-----------------------

See :doc:`../reference/utilities` for the API reference.

.. raw:: html

    <span id="astropop-math-physical-module"></span>
    <span id="astropop.math.physical.UnitsError"></span>
    <span id="astropop.math.physical.units"></span>
    <span id="classes"></span>
    <span id="functions"></span>
    <span id="module-astropop.math.physical"></span>
