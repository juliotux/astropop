
'''
imarith
-------

Handle the IRAF's imarith and imcombine functions.
'''

from ccdproc.combiner import combine
from astropy.nddata import NDArithmeticMixin

from ..logger import logger

__all__ = ['imarith', 'imcombine']


imcombine = combine


_arith_funcs = {'+': NDArithmeticMixin.add,
                '-': NDArithmeticMixin.subtract,
                '/': NDArithmeticMixin.divide,
                '*': NDArithmeticMixin.multiply}


def imarith(operand1, operand2, operation, inplace=False, logger=logger):
    """Simple arithmetic operations using CCDData.

    Supported operations: '+', '-', '*', '/'

    Keeps the header of the first image.
    """
    # TODO: manage masks
    # TODO: propagate uncertainties
    # TODO: manage caching
    logger.debug('Operation {} between {} and {}'.format(operation, operand1,
                                                         operand2))
    if operation not in _arith_funcs.keys():
        raise ValueError("Operation {} not supported.".format(operation))

    try:
        nccd = _arith_funcs[operation](operand1, operand2)
        if inplace:
            ccd = operand1
        else:
            ccd = operand1.copy()
        ccd.data = nccd.data
    except Exception as e:
        raise ValueError('Could not process the operation {} between {} and {}'
                         'Error: {}'
                         .format(operation, operand1, operand2, e))

    return nccd
