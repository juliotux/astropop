# Licensed under a 3-clause BSD style license - see LICENSE.rst
'''
imarith
-------

Handle the IRAF's imarith and imcombine functions.
'''

import numpy as np
from ccdproc.combiner import combine

from ..fits_utils import check_ccddata
from ..logger import logger

__all__ = ['imarith', 'imcombine']


# TODO: Rework combine with self implementation
imcombine = combine


_arith_funcs = {'+': np.add,
                '-': np.subtract,
                '/': np.true_divide,
                '//': np.floor_divide,
                '*': np.multiply,
                '**': np.power,
                '%': np.remainder}


# only deal with uncorrelated errors
_error_propagation = {'+': lambda sa, sb: np.sqrt(sa**2 + sb**2),
                      '-': lambda sa, sb: np.sqrt(sa**2 + sb**2),
                      '/': lambda f, a, b, sa, sb: f*np.sqrt((sa/a)**2 +
                                                             (sb/b)**2),
                      '*': lambda f, a, b, sa, sb: f*np.sqrt((sa/a)**2 +
                                                             (sb/b)**2),
                      '//': lambda f, a, b, sa, sb: f*np.sqrt((sa/a)**2 +
                                                              (sb/b)**2),
                      '**': lambda f, a, b, sa: f*b*sa/a}


def imarith(operand1, operand2, operation, inplace=False, logger=logger):
    """Simple arithmetic operations using CCDData.

    Supported operations: '+', '-', '*', '/', '**', '%', '//'

    Keeps the header of the first image.
    """
    # TODO: manage caching for results
    # TODO: handle units

    logger.debug('Operation {} between {} and {}'.format(operation, operand1,
                                                         operand2))

    operand1 = check_ccddata(operand1)
    if hasattr(operand2, 'data'):
        operand2 = check_ccddata(operand2)
        data2 = operand2.data
    else:
        data2 = operand2

    if operation not in _arith_funcs.keys():
        raise ValueError("Operation {} not supported.".format(operation))

    try:
        ndata = _arith_funcs[operation](operand1.data, data2)
    except Exception as e:
        raise ValueError('Could not process the operation {} between {} and {}'
                         'Error: {}'
                         .format(operation, operand1, operand2, e))

    # Join masks
    if hasattr(operand2, 'mask'):
        mask2 = operand2.mask
    else:
        mask2 = None
    if operand1.mask is not None and mask2 is not None:
        logger.debug('Updating mask in math operation.')
        nmask = np.logical_and(operand1.mask, mask2)
    elif operand1.mask is not None:
        nmask = operand1.mask
    else:
        nmask = None

    # propagate errors, assuming they are stddev uncertainties
    if hasattr(operand2, 'uncertainty'):
        uncertainty2 = operand2.uncertainty
    else:
        uncertainty2 = 0.0
    uncertainty2 = uncertainty2 or 0.0
    if operand1.uncertainty is not None:
        logger.debug('Propagating error in math operation')
        f = _error_propagation[operation]
        if operation in ('+', '-'):
            nuncert = f(operand1.uncertainty, uncertainty2)
        elif operation in ('*', '//', '/'):
            nuncert = f(ndata, operand1.data, data2, operand1.uncertainty,
                        uncertainty2)
        elif operation in ('**'):
            nuncert = f(ndata, operand1.data, data2, operand1.uncertainty)
        else:
            logger.warn('Operation {} does not support error propagation.'
                        .format(operation))

    if inplace:
        ccd = operand1
        ccd.data = ndata
        ccd.uncertainty = nuncert
        ccd.mask = nmask
    else:
        ccd = check_ccddata(ndata)
        ccd.meta = operand1.meta.copy()
        ccd.uncertainty = nuncert
        ccd.mask = nmask

    return ccd
