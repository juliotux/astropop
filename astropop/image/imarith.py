# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Perform arithmetic with images, physical units and uncertainties."""

from __future__ import annotations

import typing as _t
import numpy.typing as _npt
from .._typing import QuantityInput as _QuantityInput

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
    merge_flags
        ``'or'`` (default) combines FrameData flag bits by bitwise OR.
        ``'and'`` keeps only bits present in both operands.
        ``'no_merge'`` returns zero flags. Missing flags count as zero.
        Masked pixels still participate in arithmetic; flags mark the result.
    merge_headers
        ``'only_equal'`` (default) keeps keys present with equal values in both
        headers. ``'first'`` copies the first operand's header.
        ``'selected_keys'`` keeps keys named in selected_keys, taking each key's
        first occurrence. ``'no_merge'`` returns an empty header.
        Only FrameData operands contribute headers.
    selected_keys
        Header keys to keep with the 'selected_keys' strategy.

    Notes
    -----
    Instances retain only merge settings and can run multiple operations.
    Optional-array configuration is evaluated for each call to `run`.
    All calls use the same NumPy calculation on normalized values and units.
    Uncertainties use the existing derivative helpers when enabled; disabling
    them skips extraction and propagation. Floor division and remainder
    retain the existing raw-value conventions.

    Examples
    --------
    >>> from astropop.framedata import FrameData
    >>> from astropop.image import ImArith
    >>> arithmetic = ImArith(merge_headers='first')
    >>> frame = FrameData([[10.]], unit='adu', meta={'OBJECT': 'example'})

    >>> # Reuse the configured merge policy for subsequent operations.
    >>> doubled = arithmetic.run(frame, 2, '*')
    >>> result = arithmetic.run(doubled, 5, '/')
    >>> print(result.data.tolist(), result.header['OBJECT'])
    [[4.0]] example
    """

    _operations = {'+': np.add, '-': np.subtract, '*': np.multiply,
                   '/': np.true_divide, '//': np.floor_divide,
                   '**': np.power, '%': np.remainder}
    _derivatives = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'truediv',
                    '//': 'floordiv', '**': 'pow', '%': 'mod'}

    def __init__(
        self,
        merge_flags: _t.Literal['or', 'and', 'no_merge']='or',
        merge_headers: _t.Literal['only_equal', 'first', 'selected_keys', 'no_merge']='only_equal',
        selected_keys: _t.Sequence[str] | None=None,
    ) -> None:
        """Retain flag and header merge strategies for subsequent arithmetic calls."""
        self._flag_strategy = merge_flags
        self._header_strategy = merge_headers
        self._selected_keys = selected_keys

    def _as_quantity(self, value: _QuantityInput) -> u.Quantity:
        """Normalize values and units without touching optional arrays."""
        if value is None:
            raise TypeError('value must not be None.')
        if isinstance(value, (u.UnitBase, str)):
            return u.Quantity(1.0, u.Unit(value))
        unit = getattr(value, 'unit', None)
        if isinstance(value, (u.Quantity, QFloat)):
            value = value.value
        elif hasattr(value, 'data') and hasattr(value, 'unit'):
            value = value.data
        if np.any(np.asarray(value) == None) or not np.any(np.isreal(value)):  # noqa: E711
            raise TypeError('Nominal values must be real numbers and not None.')
        return u.Quantity(np.asarray(value, dtype=float), unit, copy=False)

    @_optional('FRAMEDATA_DISABLE_UNCERTAINTY')
    def _read_uncertainty(
        self,
        source: _QuantityInput,
        unit: u.UnitBase,
    ) -> _npt.ArrayLike | None:
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

    def _convert_operands(
        self,
        operand1: _QuantityInput,
        operand2: _QuantityInput,
        operation: _t.Literal['+', '-', '*', '/', '//', '%', '**'],
    ) -> tuple[tuple[u.Quantity, _npt.ArrayLike | None], tuple[u.Quantity, _npt.ArrayLike | None]]:
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

    def _validate_operation(
        self,
        operation: _t.Literal['+', '-', '*', '/', '//', '%', '**'],
        operand2: _QuantityInput,
    ) -> None:
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
    def _result_unit(
        value1: u.Quantity,
        value2: u.Quantity,
        operation: _t.Literal['+', '-', '*', '/', '//', '%', '**'],
    ) -> u.UnitBase:
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

    @_optional('FRAMEDATA_DISABLE_UNCERTAINTY')
    def _propagate_uncertainty(
        self,
        result: u.Quantity,
        first: tuple[u.Quantity, _npt.ArrayLike | None],
        second: tuple[u.Quantity, _npt.ArrayLike | None],
        operation: _t.Literal['+', '-', '*', '/', '//', '%', '**'],
    ) -> _npt.NDArray | None:
        """Reuse the derivative formulas used by QFloat without recalculating data."""
        value1, uncertainty1 = first
        value2, uncertainty2 = second
        uncertainty = propagate_2(self._derivatives[operation], result.value,
                                  value1.value, value2.value, uncertainty1, uncertainty2)
        # Constant derivatives can leave errors smaller than a broadcast
        # result, particularly when the other operand has no uncertainty.
        return np.broadcast_to(uncertainty, result.shape)

    def _calculate(
        self,
        first: tuple[u.Quantity, _npt.ArrayLike | None],
        second: tuple[u.Quantity, _npt.ArrayLike | None],
        operation: _t.Literal['+', '-', '*', '/', '//', '%', '**'],
    ) -> tuple[u.Quantity, _npt.NDArray | None]:
        """Calculate nominal data once, then optionally propagate errors."""
        value1, _ = first
        value2, _ = second
        unit = self._result_unit(value1, value2, operation)
        data = self._operations[operation](value1.value, value2.value)
        result = u.Quantity(np.asarray(data), unit, copy=False)
        uncertainty = self._propagate_uncertainty(result, first, second, operation)
        return result, uncertainty

    @_optional('FRAMEDATA_DISABLE_FLAGS')
    def _merge_flags(
        self,
        operand1: _QuantityInput,
        operand2: _QuantityInput,
        shape: tuple[int, ...],
    ) -> _npt.NDArray | None:
        """Treat missing flags as zero without reading disabled arrays."""
        flags1 = operand1.flags if isinstance(operand1, FrameData) else None
        flags2 = operand2.flags if isinstance(operand2, FrameData) else None
        if flags1 is None:
            flags1 = np.zeros(shape, dtype=np.uint8)
        if flags2 is None:
            flags2 = np.zeros(shape, dtype=np.uint8)
        return merge_flag(flags1, flags2, method=self._flag_strategy)

    def _merge_headers(
        self,
        operand1: _QuantityInput,
        operand2: _QuantityInput,
    ) -> Header:
        """Only FrameData operands contribute headers, as in imarith."""
        header1 = operand1.header if isinstance(operand1, FrameData) else Header()
        header2 = operand2.header if isinstance(operand2, FrameData) else Header()
        return merge_header(header1, header2, method=self._header_strategy,
                            selected_keys=self._selected_keys)

    def _write_result(
        self,
        output: FrameData,
        result: u.Quantity,
        uncertainty: _npt.ArrayLike | None,
        flags: _npt.ArrayLike | None,
        header: Header,
    ) -> FrameData:
        """Assign through FrameData setters to retain memory-map behavior."""
        output.data = result.value
        output.unit = result.unit
        output.uncertainty = uncertainty
        output.flags = flags
        output.meta = header
        return output

    def run(
        self,
        operand1: _QuantityInput,
        operand2: _QuantityInput,
        operation: _t.Literal['+', '-', '*', '/', '//', '%', '**'],
        inplace: bool=False,
    ) -> FrameData:
        """Perform an operation with the operand and result contract of imarith.

        Parameters
        ----------
        operand1, operand2
            Operands. Numbers and units are also accepted.
        operation
            Arithmetic operation; powers require a dimensionless scalar.
        inplace
            Update operand1 if it is FrameData. Default: False.

        Returns
        -------
        result
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


def imarith(
    operand1: _QuantityInput,
    operand2: _QuantityInput,
    operation: _t.Literal['+', '-', '*', '/', '//', '%', '**'],
    inplace: bool=False,
    merge_flags: _t.Literal['or', 'and', 'no_merge']='or',
    merge_headers: _t.Literal['only_equal', 'first', 'selected_keys', 'no_merge']='only_equal',
    **kwargs: _t.Any,
) -> FrameData:
    """Perform image arithmetic with physical units and optional uncertainties.

    Parameters
    ----------
    operand1, operand2
        Operands. Numbers, unit objects and unit strings are also accepted.
        A unit operand represents one of that unit. Numerical values are
        normalized to floating point and follow NumPy broadcasting rules.
    operation
        Arithmetic operation. Powers require a dimensionless scalar exponent.
    inplace
        Update and return operand1 when it is a FrameData. Otherwise return a
        new FrameData. Default: False.
    merge_flags
        ``'or'`` (default) combines FrameData flag bits by bitwise OR.
        ``'and'`` keeps only bits present in both operands.
        ``'no_merge'`` returns zero flags. Missing flags count as zero.
        Masked pixels still participate in arithmetic; flags mark the result.
    merge_headers
        ``'only_equal'`` (default) keeps keys present with equal values in both
        headers. ``'first'`` copies the first operand's header.
        ``'selected_keys'`` keeps keys named in selected_keys, taking each key's
        first occurrence. ``'no_merge'`` returns an empty header.
        Only FrameData operands contribute headers.
    **kwargs
        selected_keys : list of str, optional
            Keys to retain with merge_headers='selected_keys'.

    Returns
    -------
    result
        Data and unit of the result, merged flags and header, and propagated
        standard uncertainties when enabled.

    Notes
    -----
    Addition and subtraction convert the second unit to the first; incompatible
    units raise UnitsError. Multiplication and division combine units. Floor
    division and remainder use raw numerical operands without unit conversion;
    floor division returns the quotient unit and remainder the first unit.

    Uncertainties use first-order derivatives, assuming independent operands.
    Missing uncertainties contribute zero, so enabled propagation produces an
    uncertainty array even for exact inputs. CCDData uncertainties are converted
    to standard deviations. Convert CCDData with read_framedata first to retain
    its mask and metadata.

    The process-wide AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY switch skips
    extraction and propagation. The result then has uncertainty=None, including
    an in-place result. FRAMEDATA_DISABLE_FLAGS omits output flags and masks.
    Set these class attributes before loading or creating frames to avoid
    allocating optional arrays. They are evaluated at call time, but disabling
    storage after loading hides existing arrays without freeing their memory.
    Non-inplace calls leave source arrays unchanged.
    Images are not aligned, and WCS is not propagated to a new result.

    See Also
    --------
    ImArith : Reuse flag and header merge settings across operations.

    Examples
    --------
    >>> from astropy import units as u
    >>> from astropop.framedata import FrameData
    >>> from astropop.image import imarith
    >>> frame = FrameData([[1., 2.]], unit='m', uncertainty=0.1)

    >>> # Addition converts the second operand to the first operand's unit.
    >>> result = imarith(frame, 50 * u.cm, '+')
    >>> print(result.data.tolist(), result.unit)
    [[1.5, 2.5]] m
    >>> print(result.uncertainty.tolist(), frame.data.tolist())
    [[0.1, 0.1]] [[1.0, 2.0]]
    """
    arithmetic = ImArith(merge_flags=merge_flags, merge_headers=merge_headers,
                         selected_keys=kwargs.get('selected_keys', None))
    return arithmetic.run(operand1, operand2, operation, inplace=inplace)
