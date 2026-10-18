# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Perform math operations with astronomical images in FrameData contianer."""

import numpy as np
from astropy.units.core import UnitConversionError
from astropy import units as u

from ..config import _optional

from ._tools import merge_header
from ._tools import merge_flag
from ..framedata import FrameData
from ..math.physical import QFloat, convert_to_qfloat, UnitsError
from ..logger import logger, log_to_list


__all__ = ['imarith']


_arith_funcs = {'+': np.add,
                '-': np.subtract,
                '/': np.true_divide,
                '//': np.floor_divide,
                '*': np.multiply,
                '**': np.power,
                '%': np.remainder}


def _as_quantity(value):
    """Extract values and units without allocating or propagating errors."""
    if value is None:
        raise TypeError('value must not be None.')
    if isinstance(value, u.Quantity):
        return u.Quantity(np.asarray(value.value, dtype=float), value.unit,
                          copy=False)
    if isinstance(value, QFloat):
        return u.Quantity(np.asarray(value.nominal, dtype=float), value.unit,
                          copy=False, dtype=None)
    if isinstance(value, FrameData) or (hasattr(value, 'data') and
                                       hasattr(value, 'unit')):
        return u.Quantity(np.asarray(value.data, dtype=float), value.unit,
                          copy=False, dtype=None)
    if isinstance(value, (u.UnitBase, str)):
        return u.Quantity(1.0, u.Unit(value))
    return u.Quantity(np.asarray(value, dtype=float), copy=False, dtype=None)


def _arith_nominal(value1, value2, operation):
    """Calculate nominal results with the existing QFloat unit conventions."""
    if operation in ('//', '%'):
        # QFloat applies these operators to the supplied nominal numbers,
        # whereas Quantity converts compatible units first. Preserve the
        # existing arithmetic convention independently of propagation.
        data = _arith_funcs[operation](value1.value, value2.value)
        unit = value1.unit / value2.unit if operation == '//' else value1.unit
        return u.Quantity(np.asarray(data), unit, copy=False)
    if operation == '**' and (value2.unit != u.dimensionless_unscaled or
                              not np.isscalar(value2.value)):
        raise ValueError('Power operation requires a dimensionless scalar exponent')
    return _arith_funcs[operation](value1, value2)


@_optional('IMARITH_SKIP_UNCERTAINTY', 'FRAMEDATA_DISABLE_UNCERTAINTY')
def _arith_uncertainty(operand1, operand2, operation):
    """Propagate uncertainties using the existing QFloat arithmetic."""
    qf1 = convert_to_qfloat(operand1)
    qf2 = convert_to_qfloat(operand2)

    return _arith_funcs[operation](qf1, qf2).uncertainty


def _prepare_operands(operand1, operand2, operation):
    """Validate nominal exponents and convert operands for arithmetic."""
    if operation == '**':
        exponent = operand2.nominal if isinstance(operand2, QFloat) else operand2
        if isinstance(exponent, u.Quantity):
            exponent = exponent.value
        if not isinstance(exponent, (u.UnitBase, str)) and not np.isscalar(exponent):
            raise ValueError('Power operation requires a dimensionless scalar exponent')
    return _as_quantity(operand1), _as_quantity(operand2)


@_optional('FRAMEDATA_DISABLE_FLAGS')
def _merge_arithmetic_flags(operand1, operand2, shape, method):
    """Merge operand flags, honoring disabled storage and missing arrays."""
    f1 = operand1.flags if isinstance(operand1, FrameData) else None
    f2 = operand2.flags if isinstance(operand2, FrameData) else None
    if f1 is None:
        f1 = np.zeros(shape, dtype=np.uint8)
    if f2 is None:
        f2 = np.zeros(shape, dtype=np.uint8)
    return merge_flag(f1, f2, method=method)


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
    if operation not in _arith_funcs.keys():
        raise ValueError(f"Operation {operation} not supported.")

    value1, value2 = _prepare_operands(operand1, operand2, operation)

    if isinstance(operand1, FrameData) and inplace:
        ccd = operand1
    else:
        ccd = object.__new__(FrameData)

    # Add the operation entry to the ccd history.
    lh = log_to_list(logger, ccd.history)
    try:
        # Logging a QFloat itself can read its uncertainty via __str__.
        logger.debug('Operation %s between %s and %s',
                     operation, value1, value2)
        # Perform data, mask and uncertainty operations
        try:
            result = _arith_nominal(value1, value2, operation)
            uncertainty = _arith_uncertainty(operand1, operand2, operation)
        except UnitConversionError:
            raise UnitsError(f'Units {value1.unit} and {value2.unit} are'
                             f' incompatible for {operation} operation.')

        ccd.data = result.value
        ccd.unit = result.unit
        ccd.uncertainty = uncertainty

        ccd.flags = _merge_arithmetic_flags(
            operand1, operand2, ccd.shape, merge_flags)

        # Only FrameData operands contribute headers, as before conversion.
        h1 = operand1.header if isinstance(operand1, FrameData) else None
        h2 = operand2.header if isinstance(operand2, FrameData) else None
        keys = kwargs.get('selected_keys', None)
        ccd.meta = merge_header(h1, h2, method=merge_headers, selected_keys=keys)

        return ccd
    finally:
        logger.removeHandler(lh)
