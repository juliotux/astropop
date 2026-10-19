# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Stack and combine astronomical images in FrameData."""

from tempfile import mkdtemp
import numpy as np

from ..framedata import FrameData, PixelMaskFlags, check_framedata
from ..py_utils import check_number
from ..config import AstropopConfig as conf, _optional
from ..logger import logger
from ._tools import merge_header
from ._combine import _divide, _sqrt, _reduce, _statistic


__all__ = ['imcombine', 'ImCombiner']


def _sigma_clip(data, threshold=3, cen_func='median', dev_func='mad_std',
                axis=None, valid=None, block_pixels=1024):
    """Create a mask of the sigma clipped pixels.

    This function will not change the array, instead, just output a
    mask for the masked elements.

    Parameters
    ----------
    data: array_like
        Data to be clipped. Can have any dimensionality.
    threshold: int, float or tuple (optional)
        Threshold of sigma clipping. If a number is provided, it is applied
        for both low and high values. If a number is provided, it is
        interpreted as (sigma_low, sigma_high) values.
        Default: 3
    cen_func: callable or {'mean', 'median'} (optional)
        Function to compute the center value used for sigma clipping.
        Default: 'median'
    dev_func: callable or {'std', 'mad_std'} (optional)
        Function to compute the base deviation value used for sigma clipping.
        Default: 'mad_std'
    axis: int or None
        Axis along which to calculate clipping statistics.
    valid: array_like or None
        Boolean mask whose True entries identify accepted input samples.
        If None, input NaNs are excluded.
    block_pixels: int
        Maximum output pixels per statistics work block. Default: 1024.

    Returns
    -------
    mask: `~numpy.ndarray`
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
        mask |= data < cen - data.dtype.type(slow) * dev
    if shigh is not None:
        mask |= data > cen + data.dtype.type(shigh) * dev

    logger.debug('Rejected %i pixels by sigmaclip method.',
                 np.sum(mask))

    return mask


def _minmax_clip(data, min_clip=None, max_clip=None):
    """Create a mask of pixels clipped between min_clip and max_clip vals.

    Parameters
    ----------
    data: array_like
        Data array to be cliped.
    min_clip: `float`
        Minimum value accepted in the array. Values lower then this will be
        masked. `None` will disable minimum clip.
        Default: `None`
    max_clip: `float`
        Maximum value accepted in the array. Values greater then this will be
        masked. `None` will disable maximum clip.
        Default: `None`

    Returns
    -------
    mask: `~numpy.ndarray`
        Array with the same shape of `data` containing the mask for elements.
    """
    data = np.asarray(data)
    mask = ~np.isfinite(data)
    if min_clip is not None:
        mask |= data < data.dtype.type(min_clip)
    if max_clip is not None:
        mask |= data > data.dtype.type(max_clip)

    logger.debug('Rejected %i pixels by minmax method.',
                 np.sum(mask))

    return mask


def _yield_slices(shape, n_chunks):
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
    max_memory : int or float, optional
        Positive, finite budget in bytes for combination chunks and statistics
        work blocks. Internal input copies, outputs and some temporaries require
        additional memory. Default: 1e9.
    dtype : numpy dtype or str, optional
        Real numeric dtype for pixel working arrays, combination arithmetic,
        uncertainty calculations and output. Accepts NumPy integer and
        floating-point types, dtype objects and strings. Default: numpy.float64.
    tmp_dir : str, optional
        Dedicated disk-cache directory. Cleanup may remove the directory and
        its contents. If absent, create a temporary directory when disk caching
        is enabled.
    use_disk_cache : bool, optional
        Store internal input copies in memory-mapped files. Caller-owned arrays
        remain allocated. Default: False.
    **kwargs
        merge_header : {'no_merge', 'first', 'only_equal', 'selected_keys'}, optional
            Header merge strategy. Default: 'no_merge'.
        merge_header_keys : list of str, optional
            Keywords to retain with merge_header='selected_keys'.

    Notes
    -----
    Clipping is disabled initially. Configure rejection with set_sigma_clip
    and set_minmax_clip, then call combine. Settings can be reused for multiple
    collections; input copies and work arrays are released after each call,
    including failures. Optional-array configuration is evaluated at call time.

    See Also
    --------
    imcombine : Convenience function including clipping and header options.
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

    def __init__(self, max_memory=1e9, dtype=np.float64, tmp_dir=None,
                 use_disk_cache=False, **kwargs):
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

    def set_sigma_clip(self, sigma_limits=None,
                       center_func='median', dev_func='mad_std'):
        """Configure one-pass sigma rejection along the image stack.

        Parameters
        ----------
        sigma_limits : float, tuple or None, optional
            Number of deviations below and above the center. A scalar sets both
            limits; a pair sets (low, high), with None leaving a side open.
            None disables sigma rejection. Default: None.
        center_func : callable or {'median', 'mean'}, optional
            Center estimator. Default: 'median'.
        dev_func : callable or {'std', 'mad_std'}, optional
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

    def set_minmax_clip(self, min_value=None, max_value=None):
        """Configure rejection outside an accepted pixel-value range.

        Parameters
        ----------
        min_value : number or None, optional
            Reject values below this bound. None disables the lower bound.
            Default: None.
        max_value : number or None, optional
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

    def set_merge_header(self, strategy, keys=None):
        """Configure metadata merging for the combined image.

        Parameters
        ----------
        strategy : {'no_merge', 'first', 'only_equal', 'selected_keys'}
            Start with an empty header, copy the first header, retain only keys
            equal in every header, or select named keys by first occurrence.
        keys : list of str or None, optional
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

    def _clear(self):
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

    def _load_images(self, image_list, method=None):
        """Read images to FrameData and enable memmap."""
        # clear the buffers before load images.
        self._clear()

        if len(image_list) == 0:
            raise ValueError('Image list is empty.')
        if self._dtype.kind in 'iu' and len(image_list) > np.iinfo(self._dtype).max:
            raise ValueError('Image count does not fit the integer dtype; '
                             'choose a wider dtype.')

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

    def _check_consistency(self):
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

    def _block_pixels(self):
        """Bound reduction workspace independently of the image tile size."""
        # Statistics copy small groups of output pixels into contiguous sample
        # blocks. Account for one value and validity byte per image, plus an
        # allowance for per-pixel statistics, leaving half the budget for tiles.
        per_pixel = len(self._images) * (self._dtype.itemsize + 1) + 128
        return max(1, min(1024, int(self._max_memory // (2 * per_pixel))))

    def _chunk_slices(self, propagate_errors):
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

    def _read_chunk(self, slc, propagate_errors):
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

    def _chunk_yielder(self, method):
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

    def _apply_rejection(self):
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
    def _combine_uncertainty(self, method, count):
        """Compute errors separately, using the nominal values' validity."""
        if method == 'sum' and self._unct_bf is not None:
            np.square(self._unct_bf, out=self._unct_bf, where=self._valid)
            return _sqrt(_reduce(self._unct_bf, self._valid, 'sum', count))
        uncertainty = _reduce(self._buffer, self._valid, 'std', count,
                              self._block_pixels())
        root_count = _sqrt(np.asarray(count, dtype=self._dtype))
        if method == 'sum':
            uncertainty *= root_count
        else:
            _divide(uncertainty, np.maximum(root_count, 1))
        return uncertainty

    def _combine(self, method, **kwargs):
        """Coordinate nominal reduction, optional errors and normalization."""
        count = np.count_nonzero(self._valid, axis=0)
        data = _reduce(self._buffer, self._valid, method, count, self._block_pixels())
        uncertainty = self._combine_uncertainty(method, count)
        if method == 'sum' and kwargs.get('sum_normalize', True):
            norm = np.full(data.shape, len(self._buffer), dtype=self._dtype)
            _divide(norm, np.maximum(count, 1))
            data *= norm
            if uncertainty is not None:
                uncertainty *= norm
        empty = count == 0
        fill = np.nan if self._dtype.kind == 'f' else 0
        data[empty] = fill
        if uncertainty is not None:
            uncertainty[empty] = fill
        return data, uncertainty, empty | np.isnan(data)

    def combine(self, image_list, method, **kwargs):
        """Combine images using the configured clipping, dtype and header settings.

        Parameters
        ----------
        image_list : list or tuple
            Nonempty sequence of images accepted by read_framedata. All images
            must have the same shape and equal units. Source arrays are unchanged.
        method : {'mean', 'median', 'sum'}
            Combination method. See imcombine for the uncertainty estimators.
        **kwargs
            sum_normalize : bool, optional
                Multiply summed data and uncertainty by total image count divided
                by accepted sample count at each pixel. False returns the raw sum.
                Applies only to method='sum'. Default: True.

        Returns
        -------
        combined : `~astropop.framedata.FrameData`
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

            for self._buffer, self._unct_bf, slc in self._chunk_yielder(method):
                # Both rejection rules use the original tile values.
                self._apply_rejection()

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

            return combined
        finally:
            self._clear()


def imcombine(frames, method='median', memory_limit=1e9, dtype=np.float64, **kwargs):
    """Combine images with optional rejection and uncertainty estimation.

    Parameters
    ----------
    frames : list or tuple
        Nonempty sequence of images accepted by read_framedata, including
        FrameData, FITS paths or image HDUs, CCDData, Quantity, QFloat, and NumPy
        arrays. Images must have equal shapes and units; equivalent units are
        not converted automatically. Source arrays are not modified.
    method : {'median', 'mean', 'sum'}, optional
        Combination method. Default: 'median'.
    memory_limit : int or float, optional
        Positive, finite budget in bytes used to size combination chunks and
        statistics blocks. Internal input copies, output arrays and some
        temporaries are additional; this does not cap process memory.
        Default: 1e9.
    dtype : numpy dtype or str, optional
        Real numeric dtype for pixel working arrays, combination arithmetic,
        uncertainty calculations and output arrays. Accepts NumPy integer and
        floating-point types, dtype objects and strings. Its item size controls
        memory per pixel. Default: numpy.float64.
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
            Header strategy: empty header, first header, equal keys, or selected
            keys by first occurrence. Default: 'no_merge'.
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
    result : `~astropop.framedata.FrameData`
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
    and output storage. FRAMEDATA_DISABLE_FLAGS hides input masks and omits output flags; internal rejection still applies.
    Images must already be aligned; WCS is not propagated to the result.

    See Also
    --------
    ImCombiner : Reuse combination settings across image collections.
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
