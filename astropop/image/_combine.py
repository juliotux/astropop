# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Reductions on image stacks with explicit Boolean validity masks.

Stacks retain the input dtype; reductions may promote bounded working arrays.
Only owned working arrays are changed. Sample-contiguous blocks bound the
workspace needed for median and scatter calculations.
"""

from __future__ import annotations

import typing as _t
import numpy.typing as _npt
from .._typing import Axis as _Axis, Statistic as _Statistic

import numpy as np


def _pixel_blocks(
    data: _npt.NDArray,
    valid: _npt.NDArray[np.bool_],
    block_pixels: int,
) -> _t.Iterator[tuple[slice, _npt.NDArray, _npt.NDArray[np.bool_]]]:
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


def _middle_values(
    work: _npt.NDArray,
    accepted: _npt.NDArray[np.bool_],
    count: _npt.NDArray[np.intp],
) -> _npt.NDArray:
    """Select in the storage dtype, promoting only the selected middle values."""
    work[~accepted] = np.inf if work.dtype.kind == 'f' else np.iinfo(work.dtype).max
    lower = np.maximum(count - 1, 0) // 2
    upper = count // 2
    if work.dtype.kind == 'f':
        work.sort(axis=-1)
    else:
        work.partition(np.unique(np.concatenate((lower, upper))), axis=-1)
    rows = np.arange(len(work))
    # NumPy averages the selected pair using its normal reduction rules,
    # including a floating accumulator for integer samples.
    result = work[rows, lower].astype(np.result_type(work.dtype, float))
    even = (count > 0) & (count % 2 == 0)
    middle = np.stack((work[rows[even], lower[even]], work[rows[even], upper[even]]), axis=-1)
    result[even] = np.mean(middle, axis=-1)
    result[count == 0] = np.nan if work.dtype.kind == 'f' else 0
    return result


def _masked_median(
    data: _npt.NDArray,
    valid: _npt.NDArray[np.bool_],
    count: _npt.NDArray[np.intp] | None=None,
    block_pixels: int=1024,
) -> _npt.NDArray:
    """Find axis-zero medians using bounded work blocks and an explicit mask.

    True validity entries identify accepted samples, excluding input NaNs.
    Fully rejected pixels contain NaN for floating data and zero otherwise.
    Inputs are never modified. Selection uses sorting or partitioning behind
    the same storage contract. Middle-value arithmetic may promote.
    """
    if count is None:
        count = np.count_nonzero(valid, axis=0)
    dtype = np.result_type(data.dtype, float)
    result = np.empty(data.shape[1:], dtype=dtype)
    flat_result, flat_count = result.reshape(-1), count.reshape(-1)
    for slc, work, accepted in _pixel_blocks(data, valid, block_pixels):
        flat_result[slc] = _middle_values(work, accepted, flat_count[slc])
        # Release the consumer reference before the generator allocates again.
        del work, accepted
    return result


def _masked_std(
    data: _npt.NDArray,
    valid: _npt.NDArray[np.bool_],
    count: _npt.NDArray[np.intp],
    block_pixels: int,
) -> _npt.NDArray:
    """Use NumPy scatter calculations on bounded, NaN-masked work blocks."""
    result = np.full(data.shape[1:], np.nan, dtype=np.result_type(data.dtype, float))
    flat_result = result.reshape(-1)
    for slc, work, accepted in _pixel_blocks(data, valid, block_pixels):
        work = work.astype(result.dtype)
        work[~accepted] = np.nan
        nonempty = np.any(accepted, axis=-1)
        # Skip entirely rejected rows instead of asking NumPy to reduce them.
        block = np.full(len(work), np.nan, dtype=result.dtype)
        block[nonempty] = np.nanstd(work[nonempty], axis=-1)
        flat_result[slc] = block
        del work, accepted
    return result


def _masked_mad(
    data: _npt.NDArray,
    valid: _npt.NDArray[np.bool_],
    count: _npt.NDArray[np.intp],
    block_pixels: int,
) -> _npt.NDArray:
    """Gaussian-scaled MAD with NumPy handling intermediate arithmetic."""
    result = np.empty(data.shape[1:], dtype=np.result_type(data.dtype, float))
    flat_result, flat_count = result.reshape(-1), count.reshape(-1)
    for slc, work, accepted in _pixel_blocks(data, valid, block_pixels):
        work = work.astype(result.dtype)
        center = _middle_values(work.copy(), accepted, flat_count[slc])
        deviation = np.abs(work-center[:, None])
        residual_valid = accepted & ~np.isnan(deviation)
        residual_count = np.count_nonzero(residual_valid, axis=1)
        flat_result[slc] = _middle_values(deviation, residual_valid, residual_count)
        del work, accepted, deviation
    result *= 1.482602218505602
    return result


def _quadrature(
    data: _npt.NDArray,
    valid: _npt.NDArray[np.bool_],
    block_pixels: int,
) -> _npt.NDArray:
    """Use NumPy's vector norm for masked input errors, in bounded blocks."""
    result = np.empty(data.shape[1:], dtype=np.result_type(data.dtype, float))
    flat_result = result.reshape(-1)
    for slc, work, accepted in _pixel_blocks(data, valid, block_pixels):
        flat_result[slc] = np.linalg.norm(np.where(accepted, work, 0), axis=-1)
        del work, accepted
    return result


def _reduce(
    data: _npt.NDArray,
    valid: _npt.NDArray[np.bool_],
    method: _t.Literal['mean', 'median', 'sum', 'std'],
    count: _npt.NDArray[np.intp] | None=None,
    block_pixels: int=1024,
) -> _npt.NDArray:
    """Reduce nominal pixels or scatter using a shared validity mask."""
    if count is None:
        count = np.count_nonzero(valid, axis=0)
    reducers = {
        'sum': lambda: np.sum(data, axis=0, where=valid),
        'mean': lambda: np.mean(data, axis=0, where=valid),
        'median': lambda: _masked_median(data, valid, count, block_pixels),
        'std': lambda: _masked_std(data, valid, count, block_pixels),
        'mad_std': lambda: _masked_mad(data, valid, count, block_pixels),
    }
    return reducers[method]()


def _statistic(
    data: _npt.NDArray,
    valid: _npt.NDArray[np.bool_],
    func: _Statistic,
    axis: _Axis,
    block_pixels: int,
) -> _npt.NDArray:
    """Evaluate built-in statistics or protect working data from callbacks."""
    if callable(func):
        if data.dtype.kind == 'f':
            sample = data.copy()
            sample[~valid] = np.nan
        else:
            sample = np.ma.array(data, mask=~valid, copy=True)
        result = np.asarray(func(sample, axis=axis))
        if axis is not None and result.ndim == data.ndim - 1:
            result = np.expand_dims(result, axis)
        return result
    if axis is None:
        stack, accepted = data.reshape(-1, 1), valid.reshape(-1, 1)
        return _reduce(stack, accepted, func, block_pixels=block_pixels)[0]
    stack, accepted = np.moveaxis(data, axis, 0), np.moveaxis(valid, axis, 0)
    result = _reduce(stack, accepted, func, block_pixels=block_pixels)
    return np.expand_dims(result, axis)
