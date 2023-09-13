# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Perform math operations with astronomical images in FrameData contianer."""

import numpy as np
from astropy.units.core import UnitConversionError

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


def _qf_or_framedata(data, alternative=convert_to_qfloat):
    """Check if the data is QFloat or FrameData. Else, convert it."""
    if isinstance(data, (QFloat, FrameData)):
        return data
    return alternative(data)


def _arith(operand1, operand2, operation):
    """Perform the math operation itself using QFloats."""
    qf1 = convert_to_qfloat(operand1)
    qf2 = convert_to_qfloat(operand2)

    return _arith_funcs[operation](qf1, qf2)


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

    operand1 = _qf_or_framedata(operand1)
    operand2 = _qf_or_framedata(operand2)

    if isinstance(operand1, FrameData) and inplace:
        ccd = operand1
    else:
        ccd = object.__new__(FrameData)

    # Add the operation entry to the ccd history.
    lh = log_to_list(logger, ccd.history)
    logger.debug('Operation %s between %s and %s',
                 operation, operand1, operand2)

    # Perform data, mask and uncertainty operations
    try:
        result = _arith(operand1, operand2, operation)
    except UnitConversionError:
        raise UnitsError(f'Units {operand1.unit} and {operand2.unit} are'
                         f' incompatible for {operation} operation.')

    ccd.data = result.nominal
    ccd.unit = result.unit
    ccd.uncertainty = result.uncertainty

    # Perform merging flags operation only if both operands have flags
    f1 = getattr(operand1, 'flags',
                 np.zeros_like(operand1.data, dtype=np.uint8))
    f2 = getattr(operand2, 'flags',
                 np.zeros_like(operand1.data, dtype=np.uint8))
    ccd.flags = merge_flag(f1, f2, method=merge_flags)

    # Perform merging headers operation only if both operands have headers
    h1 = getattr(operand1, 'header', None)
    h2 = getattr(operand2, 'header', None)
    keys = kwargs.get('selected_keys', None)
    ccd.meta = merge_header(h1, h2, method=merge_headers, selected_keys=keys)

    logger.removeHandler(lh)
    return ccd
