
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


def imarith(operand1, operand2, operation, inplace=False, logger=logger):
    """Simple arithmetic operations using CCDData.

    Supported operations: '+', '-', '*', '/', '**', '%', '//'

    Keeps the header of the first image.
    """
    # TODO: propagate uncertainties
    # TODO: manage caching for results

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
        if inplace:
            ccd = operand1
            ccd.data = ndata
        else:
            ccd = check_ccddata(ndata)
            ccd.meta = operand1.meta.copy()
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
        operand1.mask = np.logical_and(operand1.mask, mask2)

    return ccd
