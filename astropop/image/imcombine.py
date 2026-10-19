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
    """Process the combining operation of images, like the IRAF imcombine."""

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
        """Combine images using various algorithms.

        Parameters
        ----------
        max_memory: int (optional)
          Budget in bytes used to size rejection and combination chunks.
          Input copies, output arrays and some temporaries are additional.
          Default: 1e9 (1GB)
        dtype: `~numpy.dtype` (optional)
          Real numeric dtype used consistently for pixel working arrays,
          reduction accumulators, uncertainty calculations and the result.
          Accepts NumPy integer and floating-point types, dtype objects and
          dtype strings. Controls the memory required per pixel.
          Default: `~numpy.float64`
        tmp_dir: `str` (optional)
          Directory to store temporary files used in the combining. If None,
          a tmp dir will be created with `~tempfile.mkdtemp`.
          default: `None`
        use_disk_cache: `bool`
            Cache internal image copies in memory-mapped temporary files.
            This adds disk I/O and must be enabled explicitly. Caller-owned
            frames remain in memory. Default: False.
        merge_header: {'no_merge', 'first', 'only_equal', 'selected_keys'}
            Strategy for merging headers.
            default: 'no_merge'
        merge_header_keys: `list` (optional)
            Keywords for header merging if the strategy is 'selected_keys'.
            default: None
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
        """Enable sigma clipping during the combine.

        Parameters
        ----------
        sigma_limits: `float`, `tuple` or `None` (optional)
          Set the low and high thresholds for sigma clipping. A number is
          applyed to both low and high limits. A tuple will be considered
          (low, high) limits. `None` disable the clipping.
          Default: `None`
        center_func: callable or {'median', 'mean'} (optional)
          Function to compute de central tendency of the data.
          Default: 'median'
        dev_func: callable or {'std', 'mad_std'} (optional)
          Function to compute the deviation sigma for clipping.
          Defautl: 'mad_std'

        Notes
        -----
        - 'median' and 'mad_std' gives a much better sigma clipping than
          'mean' and 'std'.
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
        """Enable minmax clipping during the combine.

        Parameters
        ----------
        min_value: `float` or `None` (optional)
          Minimum threshold of the clipping. `None` disables minimum masking.
          Default: `None`
        max_value: `float` or `None` (optional)
          Maximum threshold of the clipping. `None` disables maximum masking.
          Default: `None`
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
        """Set the strategy to merge headers during combination.

        Strategies
        ----------
        no_merge:
            No header merging will be done. The resulting combined image
            will contain only new keys based on fits standards.
            Default behavior.
        first:
            All keys from the first header will be used.
        only_equal:
            All keys are compared and just the keys that are equal between
            all hedaers are merged and kept.
        selected_keys:
            Only a list of selected keys by the user will be merged and kept.
            If they are different between headers, the value in the first
            header willbe used.

        Parameters
        ----------
        strategy: {'no_merge', 'first', 'only_equal', 'selected_keys'}
            Header merging strategy.
        keys: list
            List of the keys to be used for `selected_keys` strategy.
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
                               flags=(None if conf.FRAMEDATA_DISABLE_FLAGS
                                      else source.flags),
                               uncertainty=(self._get_uncertainty(source)
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

    @_optional('IMARITH_SKIP_UNCERTAINTY', 'FRAMEDATA_DISABLE_UNCERTAINTY')
    def _get_uncertainty(self, frame):
        """Read an input uncertainty only when propagation is enabled."""
        return frame.uncertainty

    def _block_pixels(self):
        """Bound reduction workspace independently of the image tile size."""
        per_pixel = len(self._images) * (self._dtype.itemsize + 1) + 128
        return max(1, min(1024, int(self._max_memory // (2 * per_pixel))))

    def _chunk_slices(self, propagate_errors):
        """Reserve half the workspace budget for stack data and validity."""
        n = len(self._images)
        shape = self._images[0].shape
        per_pixel = n * (self._dtype.itemsize * (1 + propagate_errors) + 1)
        # Counts and pixel results coexist with the sample stack.
        per_pixel += 4 * (np.dtype(np.intp).itemsize + self._dtype.itemsize)
        pixels = max(1, int(self._max_memory // (2 * per_pixel)))
        n_chunks = max(1, int(np.ceil(np.prod(shape) / pixels)))
        return _yield_slices(shape, n_chunks)

    def _read_chunk(self, slc, propagate_errors):
        """Copy a tile and read flag bits only for that tile."""
        shape = (len(self._images), *(s.stop - s.start for s in slc))
        data = np.empty(shape, dtype=self._dtype)
        self._valid = np.empty(shape, dtype=bool)
        uncertainty = np.empty_like(data) if propagate_errors else None
        for index, frame in enumerate(self._images):
            data[index] = frame.data[slc]
            np.isnan(data[index], out=self._valid[index])
            np.logical_not(self._valid[index], out=self._valid[index])
            flags = None if conf.FRAMEDATA_DISABLE_FLAGS else frame.flags
            if flags is not None:
                self._valid[index] &= (flags[slc] & PixelMaskFlags.MASKED.value) == 0
            if propagate_errors:
                uncertainty[index] = frame.uncertainty[slc]
                # A missing error sample contributed zero to the old nansum.
                np.copyto(uncertainty[index], 0, where=np.isnan(uncertainty[index]))
        return data, uncertainty

    def _chunk_yielder(self, method):
        """Yield owned data/error tiles and maintain their shared mask."""
        propagate = method == 'sum' and all(self._get_uncertainty(i) is not None
                                            for i in self._images)
        if method == 'sum' and not propagate:
            logger.debug('One or more frames have empty uncertainty. '
                         'Some features are disabled.')
        for slc in self._chunk_slices(propagate):
            data, uncertainty = self._read_chunk(slc, propagate)
            yield data, uncertainty, slc
            # Clear both generator and consumer references before allocation.
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

    @_optional('IMARITH_SKIP_UNCERTAINTY', 'FRAMEDATA_DISABLE_UNCERTAINTY')
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
        """Perform the image combining.

        Parameters
        ----------
        image_list: `list` or `tuple`
            List containing the images to be combined. The values in the list
            must be all of the same type and `~astropop.framedata.FrameData`
            supported.
        method: {'mean', 'median', 'sum'}
            Combining method.
        **kwargs:
            sum_normalize: bool (optional)
                If True, the imaged will be multiplied, pixel by pixel, by the
                number of images divided by the number of non-masked pixels.
                This will avoid discrepancies by different numbers of masked
                pixels across the image. If False, the raw sum of images will
                be returned.
                Default: True

        Returns
        -------
        combined: `~astropop.framedata.FrameData`
          The combined image.

        Notes
        -----
        - For now, it don't consider WCS, so it perform plain between the
          images, whitout registering.
        - Clipping parameters are set using class functions.
        - If the images exceed the maximum memory allowed, they are splited
          to perform the median and mean combine.
        - Masked elements are skiped. Result pixels will be masked if all the
          source pixels combined in it are also masked.
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
    """Combine a list of images or frames in a single one.

    Parameters
    ----------
    frames: list
        List of the frames to be combined. Can be a list of `FrameData`,
        a list of file names, a list of `~astropy.fits.ImageHDU` or
        a list of `~numpy.ndarray`. All members must have the same dimensions.
        For `FrameData`, all units must be the compatible.
    method: {'sum', 'median', 'mean'}
        Combining method. Each one has a proper math and a proper error
        computation.
    memory_limit: int (optional)
        Budget in bytes used to size rejection and combination chunks.
        Input copies, output arrays and some temporary arrays are additional;
        this is not a limit on total process memory. Default: 1e9.
    dtype: numpy dtype or str (optional)
        Real numeric dtype used consistently for pixel working arrays,
        reduction accumulators, uncertainty calculations and output arrays.
        Accepts NumPy integer and floating-point types, dtype objects and
        strings such as ``'int16'`` or ``'float32'``. Controls the memory
        required per pixel. Default: ``numpy.float64``.
    **kwargs:
        sigma_clip: float or tuple (optional)
            Threshold of sigma clipping rejection. If `None`, it disables the
            sigma clipping. If a number is provided, it is applied for both low
            and high values. If a tuple is provided, it is interpreted as
            (sigma_low, sigma_high) values.
            Default: `None`
        sigma_cen_func: callable or {'median', 'mean'} (optional)
            Function to compute the central value of sigma clipping rejection.
            If a name is provided, it must follow the convention in Notes.
            If a callable is provided, it will be applied directly on the data
            and must accept 'axis' argument.
            Default: 'median'
        sigma_dev_func: callable or {'std', 'mad_std'} (optional)
            Function to compute the std deviation of sigma clipping rejection.
            If a name is provided, it must follow the convention in Notes.
            If a callable is provided, it will be applied directly on the data
            and must accept 'axis' argument.
            Default: 'std'
        minmax_clip: tuple (optional)
            Minimum and maximum limits for minmax clipping. The values are
            interpreted as (min, max) limits. All values lower then the minimum
            limit and greater then the maximum limit will be masked. If `None`,
            the minmax clipping will be disabled.
            Default: `None`
        merge_header: {'first', 'selected_keys', 'no_merge'} (optional)
            Strategy to merge the headers of the images. If 'first', the first
            image header will be used. If 'selected_keys', the headers will be
            merged using the keys provided in the `header_merge_keys`.
            If 'no_merge', the headers will not be merged.
            Default: 'no_merge'
        header_merge_keys: list (optional)
            List of the keywords to be used to merge the headers.
            Defaut: `None`

    Returns
    -------
    result: `FrameData`
        A `FrameData` containing the combined image and its uncertainty.

    Notes
    -----
    - It is not recomended using clipping with 'sum' method, since it will
      change the number of elements to be summed in each column.
    - To disable a low or high clipping, use tuple with None. For example,
      using `sigma_clip=(None, 2)`, the lower clipping will be disabled.
    - The center function names are:
      - 'median': median of the accepted samples
      - 'mean': mean of the accepted samples
    - The deviation function names are:
      - 'std': population standard deviation of the accepted samples
      - 'mad_std': Gaussian-scaled median absolute deviation
    - Built-in statistics use the selected dtype and a shared validity mask.
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
