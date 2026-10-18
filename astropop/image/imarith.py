# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Perform arithmetic with images, physical units and uncertainties."""

import numpy as np
from astropy.units.core import UnitConversionError
from astropy import units as u
from astropy.io.fits import Header
from astropy.nddata import CCDData, StdDevUncertainty

from ..config import _optional

from ._tools import merge_header
from ._tools import merge_flag
from ..framedata import FrameData
from ..math.physical import QFloat, UnitsError
from ..math._deriv import propagate_2
from ..logger import logger, log_to_list


__all__ = ['ImArith', 'imarith']


class ImArith:
    """Configure and perform image arithmetic.

    Parameters
    ----------
    merge_flags : {'or', 'and', 'no_merge'}, optional
        Strategy for merging FrameData pixel flags. Default: 'or'.
    merge_headers : {'only_equal', 'first', 'selected_keys', 'no_merge'}, optional
        Strategy for merging FrameData headers. Default: 'only_equal'.
    selected_keys : list of str, optional
        Header keys to keep with the 'selected_keys' strategy.

    Notes
    -----
    Instances retain only merge settings and can run multiple operations.
    Optional-array configuration is evaluated for each call to `run`.
    All calls use the same NumPy calculation on normalized values and units.
    Uncertainties use the existing derivative helpers when enabled; disabling
    them skips extraction and propagation. Floor division and remainder
    retain the existing raw-value conventions.
    """

    _operations = {'+': np.add, '-': np.subtract, '*': np.multiply,
                   '/': np.true_divide, '//': np.floor_divide,
                   '**': np.power, '%': np.remainder}
    _derivatives = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'truediv',
                    '//': 'floordiv', '**': 'pow', '%': 'mod'}

    def __init__(self, merge_flags='or', merge_headers='only_equal',
                 selected_keys=None):
        self._flag_strategy = merge_flags
        self._header_strategy = merge_headers
        self._selected_keys = selected_keys

    @staticmethod
    def _nominal(value):
        """Read numerical values without reading or formatting errors."""
        return value.nominal if isinstance(value, QFloat) else value.value

    def _as_quantity(self, value):
        """Normalize values and units without touching optional arrays."""
        if value is None:
            raise TypeError('value must not be None.')
        if isinstance(value, (u.UnitBase, str)):
            return u.Quantity(1.0, u.Unit(value))
        unit = getattr(value, 'unit', None)
        if isinstance(value, (u.Quantity, QFloat)):
            value = self._nominal(value)
        elif hasattr(value, 'data') and hasattr(value, 'unit'):
            value = value.data
        if np.any(np.asarray(value) == None) or not np.any(np.isreal(value)):  # noqa: E711
            raise TypeError('Nominal values must be real numbers and not None.')
        return u.Quantity(np.asarray(value, dtype=float), unit, copy=False)

    @_optional('IMARITH_SKIP_UNCERTAINTY', 'FRAMEDATA_DISABLE_UNCERTAINTY')
    def _read_uncertainty(self, source, unit):
        """Extract standard deviations in the normalized operand unit."""
        if isinstance(source, FrameData):
            uncertainty = source.get_uncertainty(False)
        elif isinstance(source, QFloat):
            uncertainty = source.uncertainty
        elif isinstance(source, CCDData):
            uncertainty = source.uncertainty
            if uncertainty is None:
                return 0.0
            if not isinstance(uncertainty, StdDevUncertainty):
                uncertainty = uncertainty.represent_as(StdDevUncertainty)
            return uncertainty.quantity.to_value(unit)
        else:
            return 0.0
        return u.Quantity(uncertainty, source.unit).to_value(unit)

    def _convert_operands(self, operand1, operand2, operation):
        """Normalize values and optional errors before the common calculation."""
        value1 = self._as_quantity(operand1)
        value2 = self._as_quantity(operand2)
        try:
            if operation in ('+', '-'):
                value2 = value2.to(value1.unit)
            return ((value1, self._read_uncertainty(operand1, value1.unit)),
                    (value2, self._read_uncertainty(operand2, value2.unit)))
        except UnitConversionError:
            raise UnitsError(f'Units {value1.unit} and {value2.unit} are'
                             f' incompatible for {operation} operation.')

    def _validate_operation(self, operation, operand2):
        """Keep scalar-exponent validation independent of propagation."""
        if operation not in self._operations:
            raise ValueError(f'Operation {operation} not supported.')
        if operation == '**':
            exponent = operand2.nominal if isinstance(operand2, QFloat) else operand2
            if isinstance(exponent, u.Quantity):
                exponent = exponent.value
            if not isinstance(exponent, (u.UnitBase, str)) and not np.isscalar(exponent):
                raise ValueError('Power operation requires a dimensionless scalar exponent')

    @staticmethod
    def _result_unit(value1, value2, operation):
        """Apply the existing unit rules independently of error propagation."""
        if operation in ('+', '-', '%'):
            return value1.unit
        if operation in ('/', '//'):
            return value1.unit / value2.unit
        if operation == '*':
            return value1.unit * value2.unit
        if value2.unit != u.dimensionless_unscaled:
            raise ValueError('Power operation requires a dimensionless scalar exponent')
        return value1.unit ** value2.value

    @_optional('IMARITH_SKIP_UNCERTAINTY', 'FRAMEDATA_DISABLE_UNCERTAINTY')
    def _propagate_uncertainty(self, result, first, second, operation):
        """Reuse the derivative formulas used by QFloat without recalculating data."""
        value1, uncertainty1 = first
        value2, uncertainty2 = second
        uncertainty = propagate_2(self._derivatives[operation], result.value,
                                  value1.value, value2.value, uncertainty1, uncertainty2)
        # Constant derivatives can leave errors smaller than a broadcast
        # result, particularly when the other operand has no uncertainty.
        return np.broadcast_to(uncertainty, result.shape)

    def _calculate(self, first, second, operation):
        """Calculate nominal data once, then optionally propagate errors."""
        value1, _ = first
        value2, _ = second
        unit = self._result_unit(value1, value2, operation)
        data = self._operations[operation](value1.value, value2.value)
        result = u.Quantity(np.asarray(data), unit, copy=False)
        uncertainty = self._propagate_uncertainty(result, first, second, operation)
        return result, uncertainty

    @_optional('FRAMEDATA_DISABLE_FLAGS')
    def _merge_flags(self, operand1, operand2, shape):
        """Treat missing flags as zero without reading disabled arrays."""
        flags1 = operand1.flags if isinstance(operand1, FrameData) else None
        flags2 = operand2.flags if isinstance(operand2, FrameData) else None
        if flags1 is None:
            flags1 = np.zeros(shape, dtype=np.uint8)
        if flags2 is None:
            flags2 = np.zeros(shape, dtype=np.uint8)
        return merge_flag(flags1, flags2, method=self._flag_strategy)

    def _merge_headers(self, operand1, operand2):
        """Only FrameData operands contribute headers, as in imarith."""
        header1 = operand1.header if isinstance(operand1, FrameData) else Header()
        header2 = operand2.header if isinstance(operand2, FrameData) else Header()
        return merge_header(header1, header2, method=self._header_strategy,
                            selected_keys=self._selected_keys)

    def _write_result(self, output, result, uncertainty, flags, header):
        """Assign through FrameData setters to retain memory-map behavior."""
        output.data = result.value
        output.unit = result.unit
        output.uncertainty = uncertainty
        output.flags = flags
        output.meta = header
        return output

    def run(self, operand1, operand2, operation, inplace=False):
        """Perform an operation with the operand and result contract of imarith.

        Parameters
        ----------
        operand1, operand2 : FrameData, QFloat, Quantity, CCDData or array_like
            Operands. Numbers and units are also accepted.
        operation : {'+', '-', '*', '/', '//', '**', '%'}
            Arithmetic operation; powers require a dimensionless scalar.
        inplace : bool, optional
            Update operand1 if it is FrameData. Default: False.

        Returns
        -------
        FrameData
            Arithmetic result, with merged flags and headers. Non-inplace
            calls leave source data, uncertainties and flags unchanged.
        """
        self._validate_operation(operation, operand2)
        first, second = self._convert_operands(operand1, operand2, operation)
        if isinstance(operand1, FrameData) and inplace:
            output = operand1
        else:
            # The constructor requires 2D images; imarith also accepts scalars
            # and arrays of other dimensions. Populate through the setters.
            output = object.__new__(FrameData)
            output._history = []
            output._comments = []

        handler = log_to_list(logger, output.history)
        try:
            logger.debug('Operation %s between %s %s and %s %s', operation,
                         first[0].value, first[0].unit,
                         second[0].value, second[0].unit)
            result, uncertainty = self._calculate(first, second, operation)
            # Complete merges before modifying an in-place operand.
            flags = self._merge_flags(operand1, operand2, result.shape)
            header = self._merge_headers(operand1, operand2)
            return self._write_result(output, result, uncertainty, flags, header)
        finally:
            logger.removeHandler(handler)


def imarith(operand1, operand2, operation, inplace=False,
            merge_flags='or', merge_headers='only_equal', **kwargs):
    """Perform arithmetic operations using `~astropop.framedata.FrameData`.

    Notes
    -----
    * Keeps the header of the first image.

    * If ``operand1`` is not a `~astropop.framedata.FrameData` instance,
      inplies in ``inplace=False``, and a new `~astropop.framedata.FrameData`
      instance will be created.

    * Supported operations:
        - ``+`` : add. Example: 1+1=2
        - ``-`` : subtract. Example: 2-1=1
        - ``*`` : scalar product. Example: 2*3=6
        - ``/`` : true division. Example: 3/2=1.5
        - ``**`` : power. Example: 3**2=9
        - ``%`` : modulus. Example: 7%2=1
        - ``//`` : floor division. Example: 7//2=3

    Parameters
    ----------
    operand1, operand2: `~astropop.framedata.FrameData` compatible
        Values to perform the operation. `~astropop.math.physical.QFloat`
        `~astropy.units.Quantity`, numerical values and
        `~astropy.nddata.CCDData` are also suported.
    operation: {``+``, ``-``, ``*``, ``/``, ``**``, ``%``, ``//``}
        Math operation.
    inplace: bool, optional
        If True, the operations will be performed inplace in the operand 1.
    merge_flags: {'or', 'and', 'no_merge'}, optional
        How to join the masks of the operands. If ``'or'``, the resulting mask
        will be the union of the masks of the operands. If ``'and'``, the
        resulting mask will be the intersection of the masks of the operands.
        If ``'no_merge'``, the resulting mask will be only zeroes.
    merge_headers: {'no_merge', 'first', 'only_equal', 'selected_keys'}
        How to merge the headers of the operands. If ``'no_merge'``, the
        resulting header will be ``None``. If ``'first'``, the resulting
        header will be the header of the first operand. If ``'only_equal'``,
        the resulting header will be the header of the first operand, but only
        the keys that are equal in both operands. If ``'selected_keys'``, the
        resulting header will be the header of the first operand, but only the
        keys in ``selected_keys``.
    **kwargs:
        Additional arguments:
            selected_keys: list, optional
                List of keys to be merged in the header. Only used if
                ``merge_headers='selected_keys'``.

    Returns
    -------
    `~astropop.framedata.FrameData`:
        new `FrameData` instance if not ``inplace``, else the ``operand1``
        `~astropop.framedata.FrameData` instance.
    """
    arithmetic = ImArith(merge_flags=merge_flags, merge_headers=merge_headers,
                         selected_keys=kwargs.get('selected_keys', None))
    return arithmetic.run(operand1, operand2, operation, inplace=inplace)
