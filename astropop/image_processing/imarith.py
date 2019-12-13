# Licensed under a 3-clause BSD style license - see LICENSE.rst
'''
imarith
-------

Handle the IRAF's imarith and imcombine functions.
'''
# TODO: reimplement imcombine
# TODO: WCS align

import numpy as np

from ..framedata import FrameData, check_framedata
from ..logger import logger, log_to_list

__all__ = ['imarith']


_arith_funcs = {'+': np.add,
                '-': np.subtract,
                '/': np.true_divide,
                '//': np.floor_divide,
                '*': np.multiply,
                '**': np.power,
                '%': np.remainder}


def _arith_data(operand1, operand2, operation, logger):
    """Handle the arithmatics of the data."""
    # TODO: handle units
    data1 = operand1.data

    # This should be fine for numbers, nparrays, HDUs or CCDData
    if hasattr(operand2, 'data'):
        data2 = operand2.data
    else:
        data2 = operand2

    try:
        return _arith_funcs[operation](data1, data2)
    except Exception as e:
        raise ValueError(f'Could not process the operation {operation} between'
                         f'{operand1} and {operand2}. Error: {e}')


def _arith_unct(result, operand1, operand2, operation, logger):
    """Handle the arithmatics of the uncertainties."""
    def _error_propagation(f, a, b, sa, sb):
        # only deal with uncorrelated errors
        if operation in {'+', '-'}:
            return np.sqrt(sa**2 + sb**2)
        elif operation in {'*', '/', '//'}:
            return f*np.sqrt((sa/a)**2 + (sb/b)**2)
        elif operation == '**':
            return f*b*sa/a

    # TODO: handle units
    unct1 = operand1.uncertainty
    data1 = operand1.data

    if hasattr(operand2, 'uncertainty'):
        unct2 = operand2.uncertainty
    else:
        unct2 = 0.0

    if hasattr(operand2, 'data'):
        data2 = operand2.data
    else:
        data2 = operand2

    # Only propagate if operand1 has no empty uncertainty
    if not unct1.empty:
        nunct = _error_propagation(result.data, data1, data2, unct1, unct2)
    else:
        nunct = None
    return nunct


def _arith_mask(operand1, operand2, operation, logger):
    """Handle the arithmatics of the masks."""
    mask1 = operand1.mask
    # Join masks
    if hasattr(operand2, 'mask'):
        mask2 = operand2.mask
    else:
        mask2 = None

    if mask2 is not None:
        old_n = np.count_nonzero(mask1)
        nmask = np.logical_or(mask1, mask2)
        new_n = np.count_nonzero(nmask)
        logger.debug(f'Updating mask in math operation. '
                     f'From {old_n} to {new_n} masked elements.')
        return nmask
    else:
        return mask1


def _join_headers(operand1, operand2, operation, logger):
    """Join the headers to result."""
    # TODO: Think if this is the best behavior
    return operand1.header.copy()


def imarith(operand1, operand2, operation, inplace=False, logger=logger):
    """Simple arithmetic operations using CCDData.

    Notes
    -----
    * Keeps the header of the first image.

    * If `operand1` is not a `FrameData` instance, inplies in `inplace=False`,
      and a new `FrameData` instance will be created.

    * Supported operations:
        - `+` : add. Example: 1+1=2
        - `-` : subtract. Example: 2-1=1
        - `*` : scalar product. Example: 2*3=6
        - `/` : true division. Example: 3/2=1.5
        - `**` : power. Example: 3**2=9
        - `%` : modulus. Example: 7%2=1
        - `//` : floor division. Example: 7//2=3

    Parameters
    ----------
    operand1 : `FrameData` compatible
    operand2 : `FrameData` compatible, float or `astropy.units.Quantity`
    operation : {`+`, `-`, `*`, `/`, `**`, `%`, `//`}
    inplace : bool, optional
        If True, the operations will be performed inplace in the operand 1.
    logger : `logging.Logger`
        Python logger to log the actions.

    Returns
    -------
        `Framedata` : new `FrameData` instance if not `inplace`, else the
        `operand1` `FrameData` instance.
    """
    if operation not in _arith_funcs.keys():
        raise ValueError(f"Operation {operation} not supported.")

    operand1 = check_framedata(operand1)

    if inplace:
        ccd = operand1
    else:
        ccd = FrameData(None)

    lh = log_to_list(logger, ccd.history, full_record=True)
    # TODO: rewrite debug for better infos
    logger.debug(f'Operation {operation} between {operand1} and {operand2}')

    # Perform data, mask and uncertainty operations
    ccd.data = _arith_data(operand1, operand2, operation, logger)
    ccd.mask = _arith_mask(operand1, operand2, operation, logger)
    ccd.uncertainty = _arith_unct(ccd, operand1, operand2, operation, logger)
    ccd.meta = _join_headers(operand1, operand2, operation, logger)

    logger.removeHandler(lh)
    return ccd
