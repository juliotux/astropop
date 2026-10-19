# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Reductions on image stacks with explicit Boolean validity masks.

Pixel arithmetic retains the input dtype; counts and indices use np.intp.
Only owned working arrays are changed. Sample-contiguous blocks bound the
workspace needed for median and scatter calculations.
"""

import numpy as np


def _divide(data, divisor):
    """Divide owned pixels in place, preserving their dtype."""
    divisor = np.asarray(divisor, dtype=data.dtype)
    if data.dtype.kind == 'f':
        np.divide(data, divisor, out=data)
    else:
        negative_fraction = (data < 0) & (np.remainder(data, divisor) != 0)
        np.floor_divide(data, divisor, out=data)
        data[negative_fraction] += 1
    return data


def _sqrt(data):
    """Take roots in the pixel dtype without floating integer intermediates."""
    if data.dtype.kind == 'f':
        return np.sqrt(data, out=data)
    if np.any(data < 0):
        raise ValueError('Integer overflow produced a negative variance; '
                         'choose a wider dtype.')
    remainder = data.copy()
    root = np.zeros_like(data)
    # Restoring integer square root, processing two bits per iteration.
    for shift in range((data.dtype.itemsize * 8 - 1) // 2 * 2, -1, -2):
        bit = data.dtype.type(1 << shift)
        trial = root + bit
        accept = remainder >= trial
        remainder[accept] -= trial[accept]
        root >>= 1
        root[accept] += bit
    return root


def _pixel_blocks(data, valid, block_pixels):
    """Yield owned sample-contiguous data and validity blocks."""
    # Combiner tiles are C-contiguous. Two-dimensional transpose views used
    # for MAD also reshape without copying their full sample stack.
    samples = data.reshape(data.shape[0], -1)
    accepted = valid.reshape(valid.shape[0], -1)
    for start in range(0, samples.shape[1], block_pixels):
        stop = min(start + block_pixels, samples.shape[1])
        work = np.array(samples[:, start:stop].T, order='C', copy=True)
        keep = np.array(accepted[:, start:stop].T, order='C', copy=True)
        yield slice(start, stop), work, keep
        del work, keep


def _middle_values(work, accepted, count):
    """Select median values from an owned block without promoting pixels."""
    work[~accepted] = np.inf if work.dtype.kind == 'f' else np.iinfo(work.dtype).max
    lower = np.maximum(count - 1, 0) // 2
    upper = count // 2
    if work.dtype.kind == 'f':
        work.sort(axis=-1)
    else:
        work.partition(np.unique(np.concatenate((lower, upper))), axis=-1)
    rows = np.arange(len(work))
    result = work[rows, lower]
    even = (count > 0) & (count % 2 == 0)
    pair = result[even] + work[rows[even], upper[even]]
    result[even] = _divide(pair, 2)
    result[count == 0] = np.nan if result.dtype.kind == 'f' else 0
    return result


def _masked_median(data, valid, count=None, block_pixels=1024):
    """Find axis-zero medians using bounded work blocks and an explicit mask.

    True validity entries identify accepted samples, excluding input NaNs.
    Fully rejected pixels contain NaN for floating data and zero otherwise.
    Inputs are never modified. Selection uses sorting or partitioning behind
    the same dtype-preserving reduction contract.
    """
    if count is None:
        count = np.count_nonzero(valid, axis=0)
    result = np.empty(data.shape[1:], dtype=data.dtype)
    flat_result, flat_count = result.reshape(-1), count.reshape(-1)
    for slc, work, accepted in _pixel_blocks(data, valid, block_pixels):
        flat_result[slc] = _middle_values(work, accepted, flat_count[slc])
        # Release the consumer reference before the generator allocates again.
        del work, accepted
    return result


def _masked_mean(data, valid, count):
    """Compute means with explicitly typed accumulators."""
    result = np.asarray(np.sum(data, axis=0, where=valid, dtype=data.dtype.type))
    _divide(result, np.maximum(count, 1))
    if data.dtype.kind == 'f':
        result[count == 0] = np.nan
    return result


def _masked_std(data, valid, count, block_pixels):
    """Population standard deviation with one owned sample block at a time."""
    center = _masked_mean(data, valid, count).reshape(-1)
    flat_count = count.reshape(-1)
    result = np.empty(data.shape[1:], dtype=data.dtype)
    flat_result = result.reshape(-1)
    for slc, work, accepted in _pixel_blocks(data, valid, block_pixels):
        np.subtract(work, center[slc, None], out=work, where=accepted)
        np.square(work, out=work, where=accepted)
        variance = np.sum(work, axis=-1, where=accepted, dtype=data.dtype.type)
        _divide(variance, np.maximum(flat_count[slc], 1))
        flat_result[slc] = _sqrt(variance)
        del work, accepted
    if data.dtype.kind == 'f':
        result[count == 0] = np.nan
    return result


def _masked_mad(data, valid, count, block_pixels):
    """Gaussian-scaled median absolute deviation with bounded workspace."""
    center = _masked_median(data, valid, count, block_pixels).reshape(-1)
    result = np.empty(data.shape[1:], dtype=data.dtype)
    flat_result = result.reshape(-1)
    for slc, work, accepted in _pixel_blocks(data, valid, block_pixels):
        np.subtract(work, center[slc, None], out=work, where=accepted)
        np.absolute(work, out=work, where=accepted)
        residual_valid = accepted & ~np.isnan(work)
        residual_count = np.count_nonzero(residual_valid, axis=1)
        flat_result[slc] = _middle_values(work, residual_valid, residual_count)
        del work, accepted
    result *= data.dtype.type(1.482602218505602)
    return result


def _reduce(data, valid, method, count=None, block_pixels=1024):
    """Reduce nominal pixels or scatter using a shared validity mask."""
    if count is None:
        count = np.count_nonzero(valid, axis=0)
    reducers = {
        'sum': lambda: np.sum(data, axis=0, where=valid, dtype=data.dtype.type),
        'mean': lambda: _masked_mean(data, valid, count),
        'median': lambda: _masked_median(data, valid, count, block_pixels),
        'std': lambda: _masked_std(data, valid, count, block_pixels),
        'mad_std': lambda: _masked_mad(data, valid, count, block_pixels),
    }
    return reducers[method]()


def _statistic(data, valid, func, axis, block_pixels):
    """Evaluate built-in statistics or protect working data from callbacks."""
    if callable(func):
        if data.dtype.kind == 'f':
            sample = data.copy()
            sample[~valid] = np.nan
        else:
            sample = np.ma.array(data, mask=~valid, copy=True)
        result = np.asarray(func(sample, axis=axis), dtype=data.dtype)
        if axis is not None and result.ndim == data.ndim - 1:
            result = np.expand_dims(result, axis)
        return result
    if axis is None:
        stack, accepted = data.reshape(-1, 1), valid.reshape(-1, 1)
        return _reduce(stack, accepted, func, block_pixels=block_pixels)[0]
    stack, accepted = np.moveaxis(data, axis, 0), np.moveaxis(valid, axis, 0)
    result = _reduce(stack, accepted, func, block_pixels=block_pixels)
    return np.expand_dims(result, axis)
