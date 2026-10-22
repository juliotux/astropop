.. include:: ../../references.txt

.. |QFloat.uncertainty| replace:: `~astropop.math.physical.QFloat.uncertainty`
.. |QFloat.nominal| replace:: `~astropop.math.physical.QFloat.nominal`
.. |QFloat.unit| replace:: `~astropop.math.physical.QFloat.unit`

Physical Quantities and Uncertainties
=====================================

|QFloat| stores nominal values, standard uncertainties and a physical unit.
It uses Astropy units and propagates uncertainties through supported arithmetic
and NumPy operations. It is a separate container from |Quantity|, so functions
accepting Astropy quantities do not necessarily accept QFloat.

Propagation assumes independent inputs and uses first-order derivatives.
Covariance is not tracked, including when the same object appears in both
operands. Near nonlinear boundaries or for large relative errors, this model
can be inadequate. See :doc:`../framedata/uncertainties` for the scientific
assumptions and the distinction between propagation and sample scatter.

The |QFloat| Class
------------------

Supply a scalar or array of nominal values, matching uncertainties and a unit.
Omitting uncertainty stores zeros; omitting the unit creates a dimensionless
quantity. A zero stored uncertainty means none was supplied in that case,
not that a measurement has infinite precision::

    >>> import numpy as np
    >>> from astropop.math import QFloat

    >>> q = QFloat(1.0, 0.001, 'm')

    >>> print('Nominal:', q.nominal)
    Nominal: 1.0
    >>> print('Standard uncertainty:', q.uncertainty)
    Standard uncertainty: 0.001
    >>> print('Unit:', q.unit)
    Unit: m

    >>> samples = QFloat([1, 2, 3], [0.1, 0.2, 0.3], 'm/s')

    >>> print('Array shape:', samples.shape)
    Array shape: (3,)
    >>> print('Default uncertainty:', QFloat(1, unit='m').uncertainty)
    Default uncertainty: 0.0

Construction helpers and array access
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``qfloat(value, uncertainty, unit)`` is the function form of the QFloat
constructor; it returns the same container. Array uncertainties must match the
nominal array shape, rather than broadcasting a scalar error. Standard errors
are stored as absolute values. Passing an existing QFloat to ``QFloat(existing)``
copies its nominal/error arrays and unit; do not supply replacement uncertainty
or unit arguments in that form. Use ``to`` for a physical unit conversion::

    >>> from astropop.math import qfloat

    >>> measured = qfloat([10, 20], [1., 2.], 'cm')
    >>> copied = QFloat(measured)
    >>> measured.nominal[0] = 11

    >>> print('Original and copied first value:', measured.nominal[0], copied.nominal[0])
    Original and copied first value: 11.0 10.0

    >>> selected = measured[1:]

    >>> print('Selected values:', selected.nominal.tolist(), str(selected.unit))
    Selected values: [20.0] cm

    >>> # Assignment converts both the value and its uncertainty to the array's unit.
    >>> measured[0] = QFloat(0.5, 0.01, 'm')

    >>> print('Assigned values:', measured.nominal.tolist())
    Assigned values: [50.0, 20.0]
    >>> print('Assigned uncertainties:', measured.uncertainty.tolist())
    Assigned uncertainties: [1.0, 2.0]

Indexing returns a QFloat carrying the selected uncertainties and unit; indexed
assignment converts compatible units and updates both arrays. ``nominal`` and
``uncertainty`` expose the stored arrays, so editing their elements edits the
measurement. The constructor copy above remains independent. Uncertainty arrays
retain their inferred NumPy dtype; use floating-point errors when indexed
assignment must preserve fractional uncertainties. For an Astropy
Quantity, explicitly pass ``quantity.value`` and ``unit=quantity.unit`` and
supply any measurement uncertainty separately.

Updating a measurement
^^^^^^^^^^^^^^^^^^^^^^

``value`` is a read-only alias for accessing ``nominal``; ``std_dev`` is a
read/write alias of ``uncertainty``. Replacing the ``nominal`` property resets
its uncertainties to zero while retaining its unit. Set the errors afterward
when replacing nominal measurements. ``shape`` and ``size`` describe the
nominal array, and ``reset`` replaces values, uncertainties and unit together::

    >>> measured.nominal = [7, 8]

    >>> print('Errors after replacing nominal:', measured.std_dev.tolist())
    Errors after replacing nominal: [0.0, 0.0]

    >>> measured.std_dev = [0.7, 0.8]

    >>> print('Values:', measured.value.tolist(), '; shape:', measured.shape, '; size:', measured.size)
    Values: [7.0, 8.0] ; shape: (2,) ; size: 2

    >>> measured.reset(1.5, 0.1, 's')

    >>> print(f'Reset measurement: {measured.nominal:.1f} +/- {measured.uncertainty:.1f} {measured.unit}')
    Reset measurement: 1.5 +/- 0.1 s

When omitted from ``reset``, uncertainty becomes zero and unit becomes
dimensionless. Setting the unit property alone changes the unit label; it is
not a numerical conversion. Use ``to`` or ``<<`` when the numerical scale must
change with the unit.

Printing and String Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sig_digits`` controls the displayed significant digits of the uncertainty.
Formatting rounds the displayed nominal value to the corresponding decimal
place; it does not round the stored arrays. Inspect the properties when a
numeric value, rather than formatted text, is required::

    >>> formatted = QFloat(1.0583225, 0.0031495756, 'cm')

    >>> print(formatted)
    1.058+-0.003 cm

    >>> formatted.sig_digits = 2

    >>> print(formatted)
    1.0583+-0.0031 cm
    >>> print('Stored nominal:', formatted.nominal)
    Stored nominal: 1.0583225

Units and Conversion
--------------------

``to(unit)`` and ``<< unit`` return a converted QFloat, scaling both nominal
values and uncertainties. Compatible units are converted automatically during
arithmetic. Incompatible conversion raises Astropy ``UnitConversionError``::

    >>> distance = QFloat(1000, 0.01, 'm')
    >>> kilometers = distance.to('km')

    >>> print(f'Distance: {kilometers.nominal:.3f} +/- {kilometers.uncertainty:.5f} {kilometers.unit}')
    Distance: 1.000 +/- 0.00001 km
    >>> print('Centimeters:', (distance << 'cm').nominal)
    Centimeters: 100000.0

    >>> from astropy.units import UnitConversionError

    >>> try:
    ...     distance.to('s')
    ... except UnitConversionError:
    ...     print('Length cannot be converted to time.')
    Length cannot be converted to time.

Math Operations and Error Propagation
-------------------------------------

For independent inputs, the propagated variance of a function is

.. math::

    \sigma_f^2 = \sum_i \left(\frac{\partial f}{\partial x_i}\right)^2\sigma_i^2.

This approximation does not retain correlations introduced by earlier
calculations. Statistical reductions use the separate estimators below.

Supported Math Operations
^^^^^^^^^^^^^^^^^^^^^^^^^

The operators ``+``, ``-``, ``*``, ``/``, ``//``, ``%`` and ``**`` are supported;
matrix multiplication is not. Addition and subtraction require compatible
units; multiplication and division form the resulting unit::

    >>> first = QFloat(1, 0.1, 'm')
    >>> second = QFloat(200, 10, 'cm')

    >>> # Convert compatible units before adding independent measurement variances.
    >>> total = first + second

    >>> print(f'Sum: {total.nominal:.1f} +/- {total.uncertainty:.3f} {total.unit}')
    Sum: 3.0 +/- 0.141 m

    >>> speed = first / QFloat(2, 0.1, 's')

    >>> print(f'Speed: {speed.nominal:.2f} +/- {speed.uncertainty:.3f} {speed.unit}')
    Speed: 0.50 +/- 0.056 m / s

Supported Numpy Array Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supported array transformations carry both nominal and uncertainty arrays.
Do not assume NumPy view sharing: transformations returning QFloat construct
new arrays. Shape queries return ordinary shape information::

    >>> reshaped = np.reshape(samples, (3, 1))

    >>> print('Reshaped nominal:', reshaped.nominal.tolist())
    Reshaped nominal: [[1.0], [2.0], [3.0]]
    >>> print('Reshaped errors:', reshaped.uncertainty.tolist())
    Reshaped errors: [[0.1], [0.2], [0.3]]

See the :ref:`NumPy support inventory <qfloat-numpy-support>` for the supported
functions. Unsupported NumPy operations must be handled explicitly by the
application; extracting nominal arrays also discards propagation and units.

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

Inspect nominal values, uncertainties and units from each reduction::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropop.math import QFloat

    >>> q = QFloat([[1, 3], [5, 7]], [[2, 2], [2, 2]], 'adu')

    >>> # Sum propagates supplied errors; mean and median below use sample scatter.
    >>> total = np.sum(q, axis=0)

    >>> print('total.nominal:', np.round(total.nominal, 6))
    total.nominal: [ 6. 10.]
    >>> print('total.uncertainty:', np.round(total.uncertainty, 6))
    total.uncertainty: [2.828427 2.828427]
    >>> print('total.unit:', total.unit)
    total.unit: adu
    >>> for reduce in (np.mean, np.median):
    ...     result = reduce(q, axis=1)
    ...     print(reduce.__name__, 'nominal:', np.round(result.nominal, 6))
    ...     print('result.uncertainty:', np.round(result.uncertainty, 6))
    ...     print('result.unit:', result.unit)
    mean nominal: [2. 6.]
    result.uncertainty: [0.707107 0.707107]
    result.unit: adu
    median nominal: [2. 6.]
    result.uncertainty: [0.707107 0.707107]
    result.unit: adu

    >>> missing = QFloat([1, 3, 8, np.nan], [100, 100, 100, 100], 'adu')

    >>> # NaN-aware reductions omit missing nominal values from the estimate.
    >>> average = np.nanmean(missing)

    >>> print('average.nominal:', np.round(average.nominal, 6))
    average.nominal: 4.0
    >>> print('average.uncertainty:', np.round(average.uncertainty, 6))
    average.uncertainty: 1.699673

    >>> middle = np.nanmedian(missing)

    >>> print('middle.nominal:', np.round(middle.nominal, 6))
    middle.nominal: 3.0
    >>> print('middle.uncertainty:', np.round(middle.uncertainty, 6))
    middle.uncertainty: 1.47196
    >>> print('average.unit:', average.unit)
    average.unit: adu
    >>> for reduce in (np.std, np.nanstd):
    ...     values = reduce(q, axis=0)
    ...     print('type(values).__name__:', type(values).__name__)
    ...     print('values:', np.round(values, 6))
    type(values).__name__: ndarray
    values: [2. 2.]
    type(values).__name__: ndarray
    values: [2. 2.]

    >>> for reduce in (np.var, np.nanvar):
    ...     print(reduce.__name__, np.round(reduce(q, axis=0), 6))
    var [4. 4.]
    nanvar [4. 4.]

Supported Numpy Universal Functions (UFuncs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QFloat supports the calls below through NumPy's ufunc interface. Only direct
calls are implemented; methods such as ``np.add.reduce`` are unsupported.
``out`` raises ``NotImplementedError``. Other keywords are forwarded to the
individual handler, so support depends on that handler. Arithmetic handlers
such as ``add`` accept no keywords: ``where`` and ``dtype`` raise ``TypeError``.
Do not assume that all NumPy ufunc options work or are silently ignored.
Use ``np.sum(q)`` for the supported quadrature-error summation. Trigonometric
functions require an angular unit; convert degrees or radians explicitly::

    >>> angle = QFloat(30, 1, 'deg')
    >>> sine = np.sin(angle)

    >>> print(f'sin(30 deg): {sine.nominal:.3f} +/- {sine.uncertainty:.3f}')
    sin(30 deg): 0.500 +/- 0.015

See the :ref:`NumPy support inventory <qfloat-numpy-support>` for all supported
ufunc names.

Trigonometric Math
^^^^^^^^^^^^^^^^^^

Direct trigonometric and hyperbolic handlers require angles in ``deg`` or
``rad`` and return dimensionless values. Other angular units must first be
converted. Inverse handlers require dimensionless unscaled inputs and return
radians. These are the implemented conventions, including the hyperbolic
functions::

    >>> inverse = np.arcsin(QFloat(0.5, 0.01))

    >>> print(f'Angle: {inverse.nominal:.3f} +/- {inverse.uncertainty:.3f} {inverse.unit}')
    Angle: 0.524 +/- 0.012 rad

Comparisons Notes
-----------------

Choose the comparison according to the question: exact stored measurements,
interval overlap, or ordering of nominal values. None is a statistical
hypothesis test.

Equality
^^^^^^^^

``==`` compares all nominal values and uncertainties after unit conversion and
returns one Boolean for the complete object, including arrays. Incompatible
units compare unequal. ``!=`` negates that result::

    >>> first = QFloat(1, 0.1, 'm')
    >>> equivalent = QFloat(100, 10, 'cm')
    >>> different_error = QFloat(1, 0.2, 'm')

    >>> print('Equivalent measurements:', first == equivalent)
    Equivalent measurements: True
    >>> print('Different uncertainty:', first == different_error)
    Different uncertainty: False
    >>> print('Different dimension:', first == QFloat(1, 0.1, 's'))
    Different dimension: False

Equality considering errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``equal_within_errors`` checks whether standard-uncertainty intervals overlap:

.. math::

    |a-b| \leq \sigma_a + \sigma_b.

It returns an elementwise result for arrays and False for incompatible units.
This interval rule uses the sum of errors, not the quadrature uncertainty of a
difference; it does not establish that two measurements have the same physical
value::

    >>> from astropop.math.physical import equal_within_errors

    >>> print('Intervals overlap:', equal_within_errors(first, QFloat(1.15, 0.1, 'm')))
    Intervals overlap: True

Inequalities
^^^^^^^^^^^^

``<``, ``<=``, ``>`` and ``>=`` compare nominal values after unit conversion,
ignoring uncertainties. Array inputs yield elementwise comparisons.
Incompatible units raise a units error::

    >>> print('Nominal ordering:', first < QFloat(115, 5, 'cm'))
    Nominal ordering: True
    >>> print('Array ordering:', (samples > QFloat(1.5, 0, 'm/s')).tolist())
    Array ordering: [False, True, True]




Reference/API
-------------

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
:doc:`index` for the different uncertainty contracts.
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


.. currentmodule:: None
