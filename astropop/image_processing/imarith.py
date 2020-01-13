# Licensed under a 3-clause BSD style license - see LICENSE.rst
'''
imarith
-------

Handle the IRAF's imarith and imcombine functions.
'''
# TODO: reimplement imcombine

import numpy as np
from astropy import units as u

from ..framedata import FrameData, check_framedata, EmptyDataError
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
    data1 = u.Quantity(operand1.data)
    data2 = u.Quantity(operand2.data)

    try:
        return _arith_funcs[operation](data1, data2)
    except Exception as e:
        raise ValueError(f'Could not process the operation {operation} between'
                         f'{operand1} and {operand2}. Error: {e}')


def _arith_unct(result, operand1, operand2, operation, logger):
    """Handle the arithmatics of the uncertainties."""
    def _extract(operand):
        # Supose data is FrameData always
        # Transform to quantity auto handles units
        d = u.Quantity(operand.data)
        du = operand.uncertainty
        try:
            du = u.Quantity(du)
        except TypeError:
            du = 0.0*d.unit
        return d, du

    def _error_propagation(f, a, b, sa, sb):
        # only deal with uncorrelated errors
        if operation in {'+', '-'}:
            return np.sqrt(sa**2 + sb**2)
        elif operation in {'*', '/', '//'}:
            return f*np.sqrt((sa/a)**2 + (sb/b)**2)
        elif operation == '**':
            return f*b*sa/a

    data1, unct1 = _extract(operand1)
    data2, unct2 = _extract(operand2)

    # Only propagate if operand1 has no empty uncertainty
    nunct = _error_propagation(result.data, data1, data2, unct1, unct2)
    return nunct


def _arith_mask(operand1, operand2, operation, logger):
    """Handle the arithmatics of the masks."""
    def _extract(operand):
        # Supose FrameData always
        # Transform to quantity auto handles units
        d = operand.mask
        return d

    mask1 = operand1.mask
    mask2 = operand2.mask

    old_n = np.count_nonzero(mask1)
    nmask = np.logical_or(mask1, mask2)
    new_n = np.count_nonzero(nmask)
    logger.debug(f'Updating mask in math operation. 'imarith
                 f'From {old_n} to {new_n} masked elements.')
    return nmask


def _join_headers(operand1, operand2, operation, logger):
    """Join the headers to result."""
    # TODO: Think if this is the best behavior
    return operand1.header.copy()


def imarith(operand1, operand2, operation, inplace=False,
            propagate_errors=False, handle_mask=False, logger=logger):
    """Simple arithmetic operations using `~astropop.framedata.FrameData`.

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
    operand1, operand2 : `~astropop.framedata.FrameData` compatible
        Values to perform the operation. `~astropy.units.Quantity`, numerical
        values and `~astropy.nddata.CCDData` are also suported.
    operation : {``+``, ``-``, ``*``, ``/``, ``**``, ``%``, ``//``}
        Math operation.
    inplace : bool, optional
        If True, the operations will be performed inplace in the operand 1.
    propagate_errors : bool, optional
        Propagate the uncertainties during the math process.
    handle_mask : bool, optional
        Join masks in the end of the operation.
    logger : `logging.Logger`
        Python logger to log the actions.

    Returns
    -------
    `~astropop.framedata.FrameData`:
        new `FrameData` instance if not ``inplace``, else the ``operand1``
        `~astropop.framedata.FrameData` instance.
    """
    if operation not in _arith_funcs.keys():
        raise ValueError(f"Operation {operation} not supported.")

    operand1 = check_framedata(operand1)
    operand2 = check_framedata(operand2)
    if operand1.data.empty or operand2.data.empty:
        raise EmptyDataError(f'Operation {operation} not permited with empty'
                             f' data containers.')

    if inplace:
        ccd = operand1
    else:
        ccd = FrameData(None)

    lh = log_to_list(logger, ccd.history, full_record=True)
    # TODO: rewrite debug for better infos
    logger.debug(f'Operation {operation} between {operand1} and {operand2}')

    # Perform data, mask and uncertainty operations
    ccd.data = _arith_data(operand1, operand2, operation, logger)
    if handle_mask:
        ccd.mask = _arith_mask(operand1, operand2, operation, logger)
    else:
        ccd.mask = False

    if propagate_errors:
        ccd.uncertainty = _arith_unct(ccd, operand1, operand2,
                                      operation, logger)
    else:
        ccd.uncertainty = None

    ccd.meta = _join_headers(operand1, operand2, operation, logger)
    # TODO: handle WCS

    logger.removeHandler(lh)
    return ccd
