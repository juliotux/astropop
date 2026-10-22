# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Stack and combine astronomical images in FrameData."""

from __future__ import annotations

import typing as _t
import numpy.typing as _npt
from .._typing import (ImageInput as _ImageInput, PathLike as _PathLike, Axis as _Axis,
                       Statistic as _Statistic)

from tempfile import mkdtemp
import numpy as np

from ..framedata import FrameData, PixelMaskFlags, check_framedata
from ..py_utils import check_number
from ..config import AstropopConfig as conf, _optional
from ..logger import logger
from ._tools import merge_header
from ._combine import _reduce, _statistic, _quadrature


__all__ = ['imcombine', 'ImCombiner']


def _sigma_clip(
    data: _npt.ArrayLike,
    threshold: float | tuple[float | None, float | None]=3,
    cen_func: _Statistic='median',
    dev_func: _Statistic='mad_std',
    axis: _Axis=None,
    valid: _npt.NDArray[np.bool_] | None=None,
    block_pixels: int=1024,
) -> _npt.NDArray[np.bool_]:
    """Create a mask of the sigma clipped pixels.

    This function will not change the array, instead, just output a
    mask for the masked elements.

    Parameters
    ----------
    data
        Data to be clipped. Can have any dimensionality.
    threshold
        Threshold of sigma clipping. If a number is provided, it is applied
        for both low and high values. If a number is provided, it is
        interpreted as (sigma_low, sigma_high) values.
        Default: 3
    cen_func
        Function to compute the center value used for sigma clipping.
        Default: 'median'
    dev_func
        Function to compute the base deviation value used for sigma clipping.
        Default: 'mad_std'
    axis
        Axis along which to calculate clipping statistics.
    valid
        Boolean mask whose True entries identify accepted input samples.
        If None, input NaNs are excluded.
    block_pixels
        Maximum output pixels per statistics work block. Default: 1024.

    Returns
    -------
    mask
        Array with the same shape of `data` containing the mask for elements.
    """
    data = np.asarray(data)
    if valid is None:
        valid = ~np.isnan(data)
    if check_number(threshold):
        slow, shigh = threshold, threshold
    elif not np.isscalar(threshold):
        slow, shigh = threshold
    else:
        raise TypeError(f'Sigma clipping threshold {threshold} not recognized.')

    cen = _statistic(data, valid, cen_func, axis, block_pixels)
    dev = _statistic(data, valid, dev_func, axis, block_pixels)
    mask = ~np.isfinite(data)
    if slow is not None:
        mask |= data < cen - slow * dev
    if shigh is not None:
        mask |= data > cen + shigh * dev

    logger.debug('Rejected %i pixels by sigmaclip method.',
                 np.sum(mask))

    return mask


def _minmax_clip(
    data: _npt.ArrayLike,
    min_clip: float | None=None,
    max_clip: float | None=None,
) -> _npt.NDArray[np.bool_]:
    """Create a mask of pixels clipped between min_clip and max_clip vals.

    Parameters
    ----------
    data
        Data array to be cliped.
    min_clip
        Minimum value accepted in the array. Values lower then this will be
        masked. `None` will disable minimum clip.
        Default: `None`
    max_clip
        Maximum value accepted in the array. Values greater then this will be
        masked. `None` will disable maximum clip.
        Default: `None`

    Returns
    -------
    mask
        Array with the same shape of `data` containing the mask for elements.
    """
    data = np.asarray(data)
    mask = ~np.isfinite(data)
    if min_clip is not None:
        mask |= data < min_clip
    if max_clip is not None:
        mask |= data > max_clip

    logger.debug('Rejected %i pixels by minmax method.',
                 np.sum(mask))

    return mask


def _yield_slices(
    shape: tuple[int, int],
    n_chunks: int,
) -> _t.Iterator[tuple[slice, slice]]:
    """Yield slices for a given shape and number of chunks."""
    if n_chunks == 1:
        # values ust be specified for later in code
        yield slice(0, shape[0]), slice(0, shape[1])
    else:
        y_shp, x_shp = shape
        # split in y (rows) first
        ystep = int(max(1, np.floor(y_shp/n_chunks)))
        ychunks = int(np.ceil(y_shp/ystep))
        if ychunks >= n_chunks:
            xstep = x_shp
        else:
            # divide in x if needed
            xchunks = int(n_chunks/ychunks) + 1
            xstep = int(max(1, np.floor(x_shp/xchunks)))
        for y in range(0, y_shp, ystep):
            for x in range(0, x_shp, xstep):
                yield (slice(y, min(y_shp, y+ystep)),
                       slice(x, min(x_shp, x+xstep)))


class ImCombiner:
    """Configure and combine aligned images with rejection and optional errors.

    Parameters
    ----------
    max_memory
        Positive, finite budget in bytes for combination chunks and statistics
        work blocks. Internal input copies, outputs and some temporaries require
        additional memory. Default: 1e9.
    dtype
        Storage dtype for input copies, image-stack buffers and output arrays.
        NumPy controls arithmetic promotion in accumulators and bounded work
        blocks. Results are cast to this dtype only after reduction. Accepts
        NumPy integer and floating-point types, dtype objects and strings.
        Default: numpy.float64.
    tmp_dir
        Dedicated disk-cache directory. Cleanup may remove the directory and
        its contents. If absent, create a temporary directory when disk caching
        is enabled.
    use_disk_cache
        Store internal input copies in memory-mapped files. Caller-owned arrays
        remain allocated. Default: False.
    **kwargs
        merge_header : {'no_merge', 'first', 'only_equal', 'selected_keys'}, optional
            ``'no_merge'`` starts with an empty header. ``'first'`` copies the
            first input header. ``'only_equal'`` keeps keys present with equal values
            in every input header. ``'selected_keys'`` keeps the requested keys,
            taking each key's first occurrence.
            Default: ``'no_merge'``.
        merge_header_keys : list of str, optional
            Keys required by ``merge_header='selected_keys'``.

    Notes
    -----
    Clipping is disabled initially. Configure rejection with set_sigma_clip
    and set_minmax_clip, then call combine. Settings can be reused for multiple
    collections; input copies and work arrays are released after each call,
    including failures. Optional-array configuration is evaluated at call time.

    See Also
    --------
    imcombine : Convenience function including clipping and header options.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.image import ImCombiner
    >>> combiner = ImCombiner(dtype='float32', max_memory=1024**2)
    >>> combiner.set_minmax_clip(max_value=20)
    >>> images = [np.full((2, 2), value) for value in (10, 12, 100)]

    >>> # Rejection settings belong to the reusable combiner.
    >>> result = combiner.combine(images, method='mean')
    >>> print(result.data[0, 0].item())
    11.0
    """

    _sigma_clip = None  # sigmaclip thresholds
    _sigma_cen_func = None  # sigmaclip central function
    _sigma_dev_func = None  # sigmaclip deviation function
    _minmax = None  # minmax clipping parameters
    _max_memory = 1e8  # max memory to be used by the combiner
    _buffer = None  # Temporary buffer to store the image
    _unct_bf = None  # Temporary buffer to store the uncertainties
    _valid = None  # True for accepted samples, shared by data and errors
    _disk_cache = False  # Enable disk caching for _images
    _images = None  # List containing the loaded images
    _methods = {'median', 'mean', 'sum'}
    _dtype = np.float64  # Internal dtype used by the combiner
    _unit = None  # Result unit
    _shape = None  # Full image shape
    _tmpdir = None  # Directory to store temporary files
    _header_merge_keys = None  # Selected header keys for header merging
    _header_strategy = 'no_merge'  # strategy for header merging

    def __init__(
        self,
        max_memory: float=1000000000.0,
        dtype: _npt.DTypeLike=np.float64,
        tmp_dir: _PathLike | None=None,
        use_disk_cache: bool=False,
        **kwargs: _t.Any,
    ) -> None:
        """Initialize combination settings; see ImCombiner for parameters.
        """
        try:
            dtype = np.dtype(dtype)
        except (TypeError, ValueError):
            raise ValueError('Only integer or float dtypes are allowed in ImCombiner.') from None
        if dtype.kind not in 'iuf':
            raise ValueError('Only integer or float dtypes are allowed in ImCombiner.')
        if not check_number(max_memory) or not np.isfinite(max_memory) or max_memory <= 0:
            raise ValueError('max_memory must be finite and positive.')
        self._tmpdir = tmp_dir
        if self._tmpdir is None and use_disk_cache:
            self._tmpdir = mkdtemp(prefix='astropop')
        self._dtype = dtype
        self._max_memory = max_memory
        # initialize empty image list
        self._images = []
        self._disk_cache = use_disk_cache
        self.set_merge_header(kwargs.pop('merge_header', 'no_merge'),
                              kwargs.pop('merge_header_keys', None))

    def set_sigma_clip(
        self,
        sigma_limits: float | tuple[float | None, float | None] | None=None,
        center_func: _Statistic='median',
        dev_func: _Statistic='mad_std',
    ) -> None:
        """Configure one-pass sigma rejection along the image stack.

        Parameters
        ----------
        sigma_limits
            Number of deviations below and above the center. A scalar sets both
            limits; a pair sets (low, high), with None leaving a side open.
            None disables sigma rejection. Default: None.
        center_func
            Center estimator. Default: 'median'.
        dev_func
            Population standard deviation or Gaussian-scaled median absolute
            deviation. Default: 'mad_std'. The imcombine convenience function
            instead defaults to 'std'.

        Notes
        -----
        Custom estimators must accept axis and handle excluded values; see
        imcombine. If minmax rejection is also enabled, sigma statistics use
        original unmasked samples before either rejection rule is applied.
        """
        if sigma_limits is None:
            # None simply disables sigma clipping
            self._sigma_clip = None
            self._sigma_cen_func = None
            self._sigma_dev_func = None
            return

        if not np.isscalar(sigma_limits):
            if len(sigma_limits) not in (1, 2):
                raise ValueError('Invalid sigma clipping thresholds'
                                 r' {sigma_limits}')

        if not callable(center_func) and \
           center_func not in ('median', 'mean'):
            raise ValueError(f"Center function {center_func} not accpeted.")

        if not callable(dev_func) and \
           dev_func not in ('std', 'mad_std'):
            raise ValueError(f"Deviation function {dev_func} not accpeted.")

        self._sigma_clip = sigma_limits
        self._sigma_cen_func = center_func
        self._sigma_dev_func = dev_func

    def set_minmax_clip(
        self,
        min_value: float | None=None,
        max_value: float | None=None,
    ) -> None:
        """Configure rejection outside an accepted pixel-value range.

        Parameters
        ----------
        min_value
            Reject values below this bound. None disables the lower bound.
            Default: None.
        max_value
            Reject values above this bound. None disables the upper bound.
            Default: None.

        Notes
        -----
        With two bounds, sort them into increasing order. Values equal to a bound
        are accepted. Setting both to None disables minmax rejection.
        """
        low, high = min_value, max_value
        # disable
        if low is None and high is None:
            self._minmax = None
            return

        for i in (low, high):
            if not check_number(i) and i is not None:
                raise ValueError(f"{i} is not compatible with min_max "
                                 "clipping")

        # check if minimum is lower then maximum
        if low is not None and high is not None:
            low, high = sorted([low, high])

        self._minmax = (low, high)

    def set_merge_header(
        self,
        strategy: _t.Literal['only_equal', 'first', 'selected_keys', 'no_merge'],
        keys: _t.Sequence[str] | None=None,
    ) -> None:
        """Configure metadata merging for the combined image.

        Parameters
        ----------
        strategy
            ``'no_merge'`` starts with an empty header. ``'first'`` copies the
            first input header. ``'only_equal'`` keeps keys present with equal values
            in every input header. ``'selected_keys'`` keeps the requested keys,
            taking each key's first occurrence.
        keys
            Required for 'selected_keys'; ignored by other strategies.
            Default: None.

        Notes
        -----
        Image count and combination method are recorded regardless of strategy.
        WCS is not propagated to the combined image.
        """
        if strategy not in {'no_merge', 'first',
                            'only_equal', 'selected_keys'}:
            raise ValueError(f'{strategy} not known.')

        if strategy == 'selected_keys' and keys is None:
            raise ValueError('No key assigned for `selected_keys` strategy.')

        self._header_strategy = strategy
        self._header_merge_keys = keys

    def _clear(self) -> None:
        """Clear buffer and images."""
        self._buffer = None
        self._unct_bf = None
        self._valid = None
        # ensure cleaning of tmp files and free memory
        for i in self._images:
            # Release arrays without loading memory maps back into RAM.
            i.data = None
            i.flags = None
            i.uncertainty = None
        self._images = []
        self._shape = None
        self._unit = None

    def _load_images(
        self,
        image_list: _t.Sequence[_ImageInput],
        method: _t.Literal['mean', 'median', 'sum'] | None=None,
    ) -> None:
        """Read images to FrameData and enable memmap."""
        # clear the buffers before load images.
        self._clear()

        if len(image_list) == 0:
            raise ValueError('Image list is empty.')

        try:
            for indx, i in enumerate(image_list):
                # before combine, copy everything to FrameData
                source = check_framedata(i, dtype=self._dtype)
                if self._dtype.kind in 'iu' and not np.all(np.isfinite(source.data)):
                    raise ValueError('Integer combining requires finite input pixels.')
                ic = FrameData(np.array(source.data, dtype=self._dtype, copy=True),
                               unit=source.unit, dtype=self._dtype,
                               flags=source.flags,
                               uncertainty=(source.uncertainty
                                            if method in (None, 'sum') else None))
                # The source header is already normalized by FrameData.
                ic._meta = source.header.copy()

                self._images.append(ic)

                # for optimization, only enable memmap when needed.
                if self._disk_cache:
                    ic.enable_memmap(filename=f'image_{indx}',
                                     cache_folder=self._tmpdir)

        except Exception:
            self._clear()
            raise

    def _check_consistency(self) -> None:
        """Check the consistency between loaded images."""
        if len(self._images) == 0:
            raise ValueError('Combiner have no images.')
        base_shape = None
        base_unit = None
        for i, v in enumerate(self._images):
            # supose self._images only have FrameData beacuse it's protected
            if i == 0:
                base_shape = v.shape
                base_unit = v.unit
            elif v.shape != base_shape:
                raise ValueError(f"Image {i} has a shape incompatible with "
                                 "the others")
            elif v.unit != base_unit:
                raise ValueError(f"Image {i} has a unit incompatible with "
                                 "the others")
        self._shape = base_shape
        self._unit = base_unit

    def _block_pixels(self) -> int:
        """Bound reduction workspace independently of the image tile size."""
        # Statistics copy small groups of output pixels into contiguous sample
        # blocks. Include promoted subtraction/squaring and median copies,
        # leaving half the budget for the storage-dtype image tiles.
        scratch = 3 * np.result_type(self._dtype, float).itemsize
        per_pixel = len(self._images) * (self._dtype.itemsize + scratch + 1) + 128
        return max(1, min(1024, int(self._max_memory // (2 * per_pixel))))

    def _chunk_slices(self, propagate_errors: bool) -> _t.Iterator[tuple[slice, slice]]:
        """Reserve half the workspace budget for stack data and validity."""
        n = len(self._images)
        shape = self._images[0].shape
        # Every output pixel needs a sample from all n images: one data value,
        # an optional uncertainty value, and one Boolean validity byte each.
        per_pixel = n * (self._dtype.itemsize * (1 + propagate_errors) + 1)
        # Allow space for counts and intermediate pixel results, which coexist
        # with the sample stack. This is an estimate, not an exact allocation.
        per_pixel += 4 * (np.dtype(np.intp).itemsize + self._dtype.itemsize)
        # Use half the budget for the tile, leaving room for reduction scratch.
        # At least one output pixel's entire sample stack must be processed.
        # Loaded input copies and full output images are outside this budget.
        pixels = max(1, int(self._max_memory // (2 * per_pixel)))
        n_chunks = max(1, int(np.ceil(np.prod(shape) / pixels)))
        # Convert the target chunk count to spatial slices shared by all inputs.
        # The row-first tiling can produce more chunks than the requested count.
        return _yield_slices(shape, n_chunks)

    def _read_chunk(
        self,
        slc: tuple[slice, slice],
        propagate_errors: bool,
    ) -> tuple[_npt.NDArray, _npt.NDArray | None]:
        """Copy a tile and read flag bits only for that tile."""
        # Axis 0 spans images; the remaining axes are this tile's rows/columns.
        # Own these buffers so rejection and uncertainty arithmetic cannot
        # modify source arrays, including memory-mapped inputs.
        shape = (len(self._images), *(s.stop - s.start for s in slc))
        data = np.empty(shape, dtype=self._dtype)
        self._valid = np.empty(shape, dtype=bool)
        uncertainty = np.empty_like(data) if propagate_errors else None
        for index, frame in enumerate(self._images):
            data[index] = frame.data[slc]
            # True means accepted. Reuse this Boolean storage to exclude NaNs;
            # clipping later updates the same mask for both data and errors.
            np.isnan(data[index], out=self._valid[index])
            np.logical_not(self._valid[index], out=self._valid[index])
            flags = frame.flags
            if flags is not None:
                # Slice flags before deriving the mask: frame.mask would create
                # a full-image Boolean array for every tile. Other flag bits
                # describe pixel status but do not exclude samples on their own.
                self._valid[index] &= (flags[slc] & PixelMaskFlags.MASKED.value) == 0
            if propagate_errors:
                uncertainty[index] = frame.uncertainty[slc]
                # Preserve quadrature's nansum convention: a NaN error adds
                # zero variance without rejecting its nominal data sample.
                np.copyto(uncertainty[index], 0, where=np.isnan(uncertainty[index]))
        return data, uncertainty

    def _chunk_yielder(
        self,
        method: _t.Literal['mean', 'median', 'sum'],
    ) -> _t.Iterator[tuple[_npt.NDArray, _npt.NDArray | None, tuple[slice, slice]]]:
        """Yield owned data/error tiles and maintain their shared mask."""
        # Only sums with errors on every input propagate those arrays. Mean,
        # median and sums with missing errors estimate uncertainty from data
        # scatter instead. Disabled propagation also makes the getter return
        # None, avoiding an uncertainty stack altogether.
        propagate = method == 'sum' and all(i.uncertainty is not None
                                          for i in self._images)
        if method == 'sum' and not propagate:
            logger.debug('One or more frames have empty uncertainty. '
                         'Some features are disabled.')
        for slc in self._chunk_slices(propagate):
            data, uncertainty = self._read_chunk(slc, propagate)
            yield data, uncertainty, slc
            # combine() consumes the tile through self._buffer/self._unct_bf.
            # Release those references and the generator's locals before
            # allocating the next tile, avoiding two live stacks at the peak.
            self._buffer = self._unct_bf = self._valid = None
            del data, uncertainty

    def _apply_rejection(self) -> None:
        """Update validity without replacing pixels or errors with sentinels."""
        if self._valid is None:
            self._valid = ~np.isnan(self._buffer)
        if self._sigma_clip is not None:
            rejected = _sigma_clip(self._buffer, self._sigma_clip,
                                   self._sigma_cen_func, self._sigma_dev_func,
                                   axis=0, valid=self._valid,
                                   block_pixels=self._block_pixels())
            self._valid &= ~rejected
        if self._minmax is not None:
            self._valid &= ~_minmax_clip(self._buffer, *self._minmax)

    @_optional('FRAMEDATA_DISABLE_UNCERTAINTY')
    def _combine_uncertainty(
        self,
        method: _t.Literal['mean', 'median', 'sum'],
        count: _npt.NDArray[np.intp],
    ) -> _npt.NDArray | None:
        """Compute errors separately, using the nominal values' validity."""
        if method == 'sum' and self._unct_bf is not None:
            return _quadrature(self._unct_bf, self._valid, self._block_pixels())
        uncertainty = _reduce(self._buffer, self._valid, 'std', count,
                              self._block_pixels())
        root_count = np.sqrt(count)
        if method == 'sum':
            uncertainty *= root_count
        else:
            uncertainty = uncertainty / np.maximum(root_count, 1)
        return uncertainty

    def _combine(
        self,
        method: _t.Literal['mean', 'median', 'sum'],
        **kwargs: _t.Any,
    ) -> tuple[_npt.NDArray, _npt.NDArray | None, _npt.NDArray[np.bool_]]:
        """Coordinate nominal reduction, optional errors and normalization."""
        count = np.count_nonzero(self._valid, axis=0)
        data = _reduce(self._buffer, self._valid, method, count, self._block_pixels())
        uncertainty = self._combine_uncertainty(method, count)
        if method == 'sum' and kwargs.get('sum_normalize', True):
            norm = len(self._buffer) / np.maximum(count, 1)
            data = data * norm
            if uncertainty is not None:
                uncertainty *= norm
        empty = count == 0
        fill = np.nan if self._dtype.kind == 'f' else 0
        data[empty] = fill
        if uncertainty is not None:
            uncertainty[empty] = fill
        invalid = empty | np.isnan(data)
        data = np.asarray(data, dtype=self._dtype)
        if uncertainty is not None:
            uncertainty = np.asarray(uncertainty, dtype=self._dtype)
        return data, uncertainty, invalid

    def combine(
        self,
        image_list: _t.Sequence[_ImageInput],
        method: _t.Literal['mean', 'median', 'sum'],
        **kwargs: _t.Any,
    ) -> FrameData:
        """Combine images using the configured clipping, dtype and header settings.

        Parameters
        ----------
        image_list
            Nonempty sequence of images accepted by read_framedata. All images
            must have the same shape and equal units. Source arrays are unchanged.
        method
            Combination method. See imcombine for the uncertainty estimators.
        **kwargs
            sum_normalize : bool, optional
                Multiply summed data and uncertainty by total image count divided
                by accepted sample count at each pixel. False returns the raw sum.
                Applies only to method='sum'. Default: True.

        Returns
        -------
        combined
            Combined image in the configured dtype and input unit, with optional
            uncertainty and mask arrays.

        Notes
        -----
        Masked samples and NaNs are excluded. Pixels with no accepted samples are
        masked when flag storage is enabled. Images must already be aligned; WCS
        is not propagated. The memory budget sizes chunks and statistics blocks;
        internal input copies and output arrays require additional memory. Internal
        arrays and mapped files are released on success and on failure. Clipping
        and header settings remain available for subsequent calls.
        """
        if method not in self._methods:
            raise ValueError(f'{method} is not a valid combining method.')

        try:
            # first of all, load the images to FrameData and check the consistency
            self._load_images(image_list, method)
            self._check_consistency()

            logger.info('Combining %i images with %s method.',
                        len(self._images), method)

            # temp combined data, mask and uncertainty
            data = np.empty(self._shape, dtype=self._dtype)
            unct = None
            mask = (None if conf.FRAMEDATA_DISABLE_FLAGS
                    else np.empty(self._shape, dtype=bool))

            filtering = self._sigma_clip is not None or self._minmax is not None
            eligible_samples = rejected_samples = 0
            for self._buffer, self._unct_bf, slc in self._chunk_yielder(method):
                if filtering:
                    candidates = np.count_nonzero(self._valid)
                # Both rejection rules use the original tile values.
                self._apply_rejection()
                if filtering:
                    eligible_samples += candidates
                    rejected_samples += candidates - np.count_nonzero(self._valid)

                # combine the images and compute the uncertainty
                data[slc], chunk_uncertainty, chunk_mask = self._combine(method, **kwargs)
                if chunk_uncertainty is not None:
                    if unct is None:
                        unct = np.empty(self._shape, dtype=self._dtype)
                    unct[slc] = chunk_uncertainty
                if mask is not None:
                    mask[slc] = chunk_mask

            n = len(self._images)
            combined = FrameData(data, unit=self._unit, dtype=self._dtype,
                                 uncertainty=unct, mask=mask)
            combined.meta = merge_header(*[i.header for i in self._images],
                                         method=self._header_strategy,
                                         selected_keys=self._header_merge_keys)
            combined.meta['HIERARCH astropop imcombine nimages'] = n
            combined.meta['HIERARCH astropop imcombine method'] = method
            if filtering:
                fraction = rejected_samples / eligible_samples if eligible_samples else 0.0
                logger.info('Combination filters rejected %i of %i initially valid '
                            'samples (%.2f%%).', rejected_samples, eligible_samples,
                            100*fraction)

            return combined
        finally:
            self._clear()


def imcombine(
    frames: _t.Sequence[_ImageInput],
    method: _t.Literal['mean', 'median', 'sum']='median',
    memory_limit: float=1000000000.0,
    dtype: _npt.DTypeLike=np.float64,
    **kwargs: _t.Any,
) -> FrameData:
    """Combine images with optional rejection and uncertainty estimation.

    Parameters
    ----------
    frames
        Nonempty sequence of images accepted by read_framedata, including
        FrameData, FITS paths or image HDUs, CCDData, Quantity, QFloat, and NumPy
        arrays. Images must have equal shapes and units; equivalent units are
        not converted automatically. Source arrays are not modified.
    method
        Combination method. Default: 'median'.
    memory_limit
        Positive, finite budget in bytes used to size combination chunks and
        statistics blocks. Internal input copies, output arrays and some
        temporaries are additional; this does not cap process memory.
        Default: 1e9.
    dtype
        Storage dtype for input copies, image-stack buffers and output arrays.
        Temporary accumulators, clipping statistics and uncertainty calculations
        may promote to wider types; the full stack stays in the requested dtype.
        NumPy controls temporary arithmetic and the final conversion, including
        integer truncation, overflow and precision limits.
        Accepts NumPy integer and floating-point types, dtype objects and strings.
        Default: numpy.float64.
    **kwargs
        sigma_clip : float or tuple, optional
            Rejection threshold in multiples of the selected deviation. A
            number applies to both sides; a pair gives (low_sigma, high_sigma).
            Use None for an open side. Default: None (disabled).
        sigma_cen_func : callable or {'median', 'mean'}, optional
            Center estimator for sigma rejection. Default: 'median'.
        sigma_dev_func : callable or {'std', 'mad_std'}, optional
            Deviation estimator: population standard deviation or
            Gaussian-scaled median absolute deviation. Default: 'std'.
        minmax_clip : tuple, optional
            (low, high) accepted pixel limits. Values outside are rejected;
            None leaves a side open. Default: None (disabled).
        sum_normalize : bool, optional
            For sums, multiply data and uncertainty by total image count divided
            by accepted sample count at each pixel. False returns the raw sum.
            Default: True.
        merge_header : {'no_merge', 'first', 'only_equal', 'selected_keys'}, optional
            ``'no_merge'`` starts with an empty header. ``'first'`` copies the
            first input header. ``'only_equal'`` keeps keys present with equal values
            in every input header. ``'selected_keys'`` keeps the requested keys,
            taking each key's first occurrence.
            Default: ``'no_merge'``.
        merge_header_keys : list of str, optional
            Keys for merge_header='selected_keys'. Default: None.
        use_disk_cache : bool, optional
            Store internal input copies in memory-mapped temporary files.
            Caller-owned arrays remain allocated. Default: False.
        tmp_dir : str, optional
            Dedicated directory for disk-cache files. Cache cleanup may remove
            the directory and its contents. When absent, a temporary directory
            is created if disk caching is enabled.

    Returns
    -------
    result
        Combined data, optional standard uncertainties and output mask, in the
        selected dtype and input unit. The header records the image count and
        combination method, in addition to any merged keys.

    Notes
    -----
    Input MASKED pixels and NaNs are excluded. Rejection uses one shared validity
    mask for data and errors. Sigma clipping makes one pass; when both rejection
    rules are enabled, sigma statistics use the original unmasked samples.
    Built-in statistics use bounded work blocks. Custom estimators must accept
    axis and handle excluded samples in a protected copy (NaNs for floating
    arrays, a masked array for integer data); their workspace is not bounded.

    Mean and median uncertainty is population scatter divided by sqrt(n), where
    n is the accepted sample count. Sum uncertainty propagates all available
    input errors in quadrature; if any input lacks errors, it uses population
    scatter times sqrt(n). Normalized sums then scale both data and uncertainty
    by N/n. Rejected errors do not contribute; NaN input errors contribute zero.
    Pixels with no accepted samples are masked when flag storage is enabled.

    AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY skips uncertainty calculations
    and output storage. FRAMEDATA_DISABLE_FLAGS hides input masks and omits
    output flags; internal rejection still applies. Set these class attributes
    before loading or creating frames to avoid allocating optional arrays.
    Disabling storage after loading hides existing arrays without freeing
    their memory.
    Images must already be aligned; WCS is not propagated to the result.

    See Also
    --------
    ImCombiner : Reuse combination settings across image collections.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.image import imcombine
    >>> images = [np.full((2, 2), value) for value in (10, 12, 100)]

    >>> # Reject samples above the accepted range before averaging.
    >>> result = imcombine(images, method='mean', minmax_clip=(None, 20),
    ...                    dtype='float32')
    >>> print(result.data.tolist(), result.dtype.name)
    [[11.0, 11.0], [11.0, 11.0]] float32
    """
    # Sanitize kwargs and create combiner
    kargs = {'dtype': dtype,
             'use_disk_cache': kwargs.pop('use_disk_cache', False),
             'tmp_dir': kwargs.pop('tmp_dir', None)}
    combiner = ImCombiner(max_memory=memory_limit, **kargs)

    sc = kwargs.pop('sigma_clip', None)
    if sc is not None:
        sigma_cen_func = kwargs.pop('sigma_cen_func', 'median')
        sigma_dev_func = kwargs.pop('sigma_dev_func', 'std')
        combiner.set_sigma_clip(sc, sigma_cen_func, sigma_dev_func)

    mm = kwargs.pop('minmax_clip', None)
    if mm is not None:
        combiner.set_minmax_clip(*mm)

    merge_header = kwargs.pop('merge_header', 'no_merge')
    header_keys = kwargs.pop('merge_header_keys', None)
    combiner.set_merge_header(merge_header, header_keys)

    # Perform the combinations
    return combiner.combine(frames, method, **kwargs)
