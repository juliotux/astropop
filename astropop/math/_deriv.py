# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Numerical and analitical math operations derivatives."""

import sys
import copy
import numpy as np
import itertools


__all__ = ['partial_derivative', 'NumericalDerivative',
           'numpy_ufunc_derivatives', 'math_derivatives']

# pylint:disable=no-else-return


def partial_derivative(f, arg_ref):
    """Calculate the partial derivative of a function numerically.

    Parameters
    ----------
    f : function
        Function to calculate the partial derivative.
    arg_ref : `string` or `int`
        Which variable to use for the differentiation. `int` implies the index
        of variable in positional args, `string` implies the name of the
        variable in kwargs.
    """
    def partial_derivative_of_f(*args, **kwargs):
        """
        Partial derivative, calculated with the (-epsilon, +epsilon)
        method, which is more precise than the (0, +epsilon) method.
        """

        keys = copy.copy(kwargs)
        for i, v in itertools.enumerate(args):
            keys[i] = v

        if isinstance(arg_ref, str):
            if arg_ref not in kwargs.keys():
                raise KeyError(f'{arg_ref} variable not in function kwargs.')
        else:
            if arg_ref >= len(args):
                raise KeyError(f'{arg_ref} index not in function args range '
                               f'{len(args)}.')

        # The step is relative to the parameter being varied, so that
        # shifting it does not suffer from finite precision limitations:
        STEP_SIZE = np.sqrt(sys.float_info.epsilon)
        step = STEP_SIZE*abs(keys[arg_ref])
        if not step:
            # Arbitrary, but "small" with respect to 1:
            step = STEP_SIZE

        # Compute f(x+eps) and f(x-eps)
        difs = {}
        for i in [-1, +1]:
            nkwargs = copy.copy(keys)
            nkwargs[arg_ref] += i*step
            nargs = [nkwargs.pop(k) for k in range(len(args))]
            difs[i] = f(*nargs, **nkwargs)

        return (difs[+1] - difs[-1])/2/step

    return partial_derivative_of_f


class NumericalDerivative:
    """Easy get partial derivatives of a function using items.

    Parameters
    ----------
    func : `callable`
        Function to get the partial derivatives.

    Examples
    --------
    To get a partial numerical derivative of a function F, just pass it to the
    initializer of this class and get the partial derivative using getitem.
    >>> func = lambda x, y : return x + y
    >>> pd = NumericalDerivative(func)
    >>> pdx = pd[0]
    >>> pdy = pd['y']
    """

    def __init__(self, func):
        if not callable(func):
            raise ValueError(f'Function {func} not callable')
        self._func = func

    def __getitem__(self, k):
        return partial_derivative(self._func, k)


def nan_except(func):
    """Return `nan` in place of raise some erros."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, ZeroDivisionError, OverflowError):
            return float('nan')
    return wrapper


def python_math_operators():
    """Return a dict of mathematical operations (x, y) partial derivatives."""
    def pow_deriv_0(x, y):
        if y == 0:
            return 0.0
        elif x != 0 or y % 1 == 0:
            return y*x**(y-1)
        else:
            return float('nan')

    def pow_deriv_1(x, y):
        if x == 0 and y > 0:
            return 0.0
        else:
            return np.log(x)*x**y

    _derivs = {
        # f(x,y)=x+y
        'add': (lambda x, y: 1.0,
                lambda x, y: 1.0),
        # f(x,y)=x-y
        'sub': (lambda x, y: 1.0,
                lambda x, y: -1.0),
        # f(x,y)=x*y
        'mul': (lambda x, y: y,
                lambda x, y: x),
        # f(x,y)=x/y
        'div': (lambda x, y: 1/y,
                lambda x, y: -x/y**2),
        # f(x,y)=x//y -> non-exact
        'floordiv': (lambda x, y: 0.0,
                     lambda x, y: 0.0),
        # f(x,y)=x%y
        'mod': (lambda x, y: 1.0,
                partial_derivative(float.__mod__, 1)),
        # f(x,y)=x/y
        'truediv': (lambda x, y: 1/y,
                    lambda x, y: -x/y**2),
        # f(x,y)=x**y
        'pow': (pow_deriv_0,
                pow_deriv_1)
    }

    # List of lambda partial derivatives of numerical python operations
    _derivs2 = {}
    for op in _derivs:
        _derivs2[f"r{op}"] = reversed(_derivs[op])
    _derivs.update(_derivs2)

    def _abs_deriv(x):
        if x > 0:
            return 1.0
        return -1.0

    _derivs.update({
        'abs': [_abs_deriv],
        'neg': [lambda x: 1.0],
        'pos': [lambda x: 1.0],
    })

    for k, v in _derivs.items():
        _derivs[k] = [np.vectorize(nan_except(w)) for w in v]

    return _derivs


def numpy_math_operators():  # noqa
    def log_der0(*args):
        if len(args) == 1:
            return 1/args[0]
        else:
            return 1/args[0]/np.log(args[1])  # 2-argument form

    def _deriv_copysign(x, y):
        if x >= 0:
            return np.copysign(1, y)
        else:
            return -np.copysign(1, y)

    def _deriv_fabs(x):
        if x >= 0:
            return 1
        else:
            return -1

    def _deriv_pow_0(x, y):
        if y == 0:
            return 0.0
        elif x != 0 or y % 1 == 0:
            return y*np.pow(x, y-1)
        else:
            return np.float('nan')

    def _deriv_pow_1(x, y):
        if x == 0 and y > 0:
            return 0.0
        else:
            return np.log(x) * np.pow(x, y)

    erf_coef = 2/np.sqrt(np.pi)  # Optimization for erf()

    _derivs = {
        # In alphabetical order, here:
        'acos': [lambda x: -1/np.sqrt(1-x**2)],
        'acosh': [lambda x: 1/np.sqrt(x**2-1)],
        'asin': [lambda x: 1/np.sqrt(1-x**2)],
        'asinh': [lambda x: 1/np.sqrt(1+x**2)],
        'atan': [lambda x: 1/(1+x**2)],
        'atan2': [lambda y, x: x/(x**2+y**2),  # Correct for x == 0
                  lambda y, x: -y/(x**2+y**2)],  # Correct for x == 0
        'atanh': [lambda x: 1/(1-x**2)],
        'copysign': [_deriv_copysign,
                     lambda x, y: 0],
        'cos': [lambda x: -np.sin(x)],
        'cosh': [np.sinh],
        'degrees': [lambda x: np.degrees(1)],
        'erf': [lambda x: np.exp(-x**2)*erf_coef],
        'erfc': [lambda x: -np.exp(-x**2)*erf_coef],
        'exp': [np.exp],
        'expm1': [np.exp],
        'fabs': [_deriv_fabs],
        'hypot': [lambda x, y: x/np.hypot(x, y),
                  lambda x, y: y/np.hypot(x, y)],
        'log': [log_der0,
                lambda x, y: -np.log(x, y)/y/np.log(y)],
        'log10': [lambda x: 1/x/np.log(10)],
        'log1p': [lambda x: 1/(1+x)],
        'pow': [_deriv_pow_0, _deriv_pow_1],
        'radians': [lambda x: np.radians(1)],
        'sin': [np.cos],
        'sinh': [np.cosh],
        'sqrt': [lambda x: 0.5/np.sqrt(x)],
        'tan': [lambda x: 1+np.tan(x)**2],
        'tanh': [lambda x: 1-np.tanh(x)**2]
    }

    for k, v in _derivs.items():
        _derivs[k] = [np.vectorize(nan_except(w)) for w in v]

    return _derivs


numpy_ufunc_derivatives = numpy_math_operators()
math_derivatives = python_math_operators()
