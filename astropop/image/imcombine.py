# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Stack and combine astronomical images in FrameData."""

import functools
from tempfile import mkdtemp
import numpy as np
from astropy.stats import mad_std

from ..framedata import FrameData, check_framedata
from ..py_utils import check_number
from ..config import AstropopConfig as conf, _optional
from ..logger import logger
from ._tools import merge_header


# bottleneck has faster median
try:
    import bottleneck as bn
    _funcs = {
        'median': bn.nanmedian,
        'mean': bn.nanmean,
        'sum': bn.nansum,
        'std': bn.nanstd
    }
except ImportError:
    _funcs = {
        'median': np.nanmedian,
        'mean': np.nanmean,
        'sum': np.nansum,
        'std': np.nanstd
    }


__all__ = ['imcombine', 'ImCombiner']


_funcs['mad_std'] = functools.partial(mad_std, ignore_nan=True)


def _integer_divide(data, divisor):
    """Divide integer pixels in place, truncating towards zero."""
    divisor = np.asarray(divisor, dtype=data.dtype)
    remainder = np.remainder(data, divisor)
    negative = data < 0
    np.floor_divide(data, divisor, out=data, casting='unsafe')
    data[negative & (remainder != 0)] += 1
    return data


def _integer_sqrt(data):
    """Integer square root without floating-point working arrays."""
    if np.any(data < 0):
        raise ValueError('Integer overflow produced a negative variance; '
                         'choose a wider dtype.')
    # Restoring square root, processing two bits per iteration. All working
    # values retain the requested integer dtype, including its signedness.
    remainder = data.copy()
    root = np.zeros_like(remainder)
    for shift in range((data.dtype.itemsize * 8 - 1) // 2 * 2, -1, -2):
        bit = remainder.dtype.type(1 << shift)
        trial = root + bit
        accept = remainder >= trial
        remainder[accept] -= trial[accept]
        root >>= 1
        root[accept] += bit
    return root


def _integer_reduce(data, mask, method):
    """Reduce integer pixels without promoting pixel accumulators.

    Counts and selection indices use NumPy's index dtype. Pixel arithmetic
    wraps in the selected dtype, and integer divisions truncate towards zero.
    Fully rejected pixels have the placeholder value zero.
    """
    count = np.count_nonzero(~mask, axis=0)
    divisor = np.maximum(count, 1)
    if method in ('sum', 'mean'):
        result = np.sum(data, where=~mask, axis=0, dtype=data.dtype.type)
        return _integer_divide(result, divisor) if method == 'mean' else result
    if method == 'median':
        ordered = np.sort(np.where(mask, np.iinfo(data.dtype).max, data), axis=0)
        lower = np.maximum(count - 1, 0) // 2
        upper = count // 2
        low = np.take_along_axis(ordered, lower[None], axis=0)[0]
        high = np.take_along_axis(ordered, upper[None], axis=0)[0]
        # Only even samples need addition; a one-sample median must not wrap.
        even = (count % 2 == 0) & (count > 0)
        low[even] = _integer_divide(low[even] + high[even], 2)
        low[count == 0] = 0
        return low
    center = _integer_reduce(data, mask, 'mean' if method == 'std' else 'median')
    delta = np.subtract(data, center, dtype=data.dtype.type)
    if method == 'std':
        np.square(delta, out=delta)
        variance = _integer_reduce(delta, mask, 'mean')
        return _integer_sqrt(variance)
    if method == 'mad_std':
        # MAD's Gaussian scale factor (1.4826...) becomes 1 in integer
        # arithmetic, just as any other fractional constant is truncated.
        np.absolute(delta, out=delta)
        return _integer_reduce(delta, mask, 'median')
    raise ValueError(f'{method} is not an integer reduction method.')


def _sigma_clip(data, threshold=3, cen_func='median', dev_func='mad_std',
                axis=None):
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
    axis: int
        The axis to perform the clipping and masking.

    Returns
    -------
    mask: `~numpy.ndarray`
        Array with the same shape of `data` containing the mask for elements.
    """
    data = np.array(data)

    if check_number(threshold):
        slow = threshold
        shigh = threshold
    elif not np.isscalar(threshold):
        slow, shigh = threshold
    else:
        raise TypeError(f'Sigma clipping threshold {threshold} not'
                        ' recognized.')

    if not callable(cen_func):
        cen_func = _funcs[cen_func]
    if not callable(dev_func):
        dev_func = _funcs[dev_func]

    cen = cen_func(data, axis=axis)
    dev = dev_func(data, axis=axis)

    # also mask nans and infs
    mask = ~np.isfinite(data)
    if slow is not None:
        mask |= data < cen-(slow*dev)
    if shigh is not None:
        mask |= data > cen+(shigh*dev)

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
    data = np.array(data)
    # masking nan and infinity
    mask = ~np.isfinite(data)

    if min_clip is not None:
        mask[np.where(data < min_clip)] = True

    if max_clip is not None:
        mask[np.where(data > max_clip)] = True

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
    _rejected = None  # Separate validity mask for integer working arrays
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
        self._tmpdir = tmp_dir
        if self._tmpdir is None:
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
        self._rejected = None
        # ensure cleaning of tmp files and free memory
        for i in self._images:
            # Release arrays without loading memory maps back into RAM.
            i.data = None
            i.flags = None
            i.uncertainty = None
        self._images = []
        self._shape = None
        self._unit = None

    def _load_images(self, image_list):
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
                               uncertainty=self._get_uncertainty(source))
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

    def _chunk_yielder(self, method):
        """Split the data in chuncks according to the method."""
        # sum needs uncertainties, others ignore it
        unct = method == 'sum' and all(self._get_uncertainty(i) is not None
                                      for i in self._images)
        if method == 'sum' and not unct:
            logger.debug('One or more frames have empty uncertainty. '
                         'Some features are disabled.')

        # flags and uncertainty are ignored
        shape = self._images[0].shape
        tot_size = self._images[0].data.nbytes
        tot_size *= len(self._images)
        # uncertainty is ignored

        # adjust memory usage for numpy and bottleneck, based on ccdproc
        if method == 'median':
            tot_size *= 4.5
        else:
            tot_size *= 3

        n_chunks = np.ceil(tot_size/self._max_memory)
        slices = list(_yield_slices(shape, n_chunks))
        if n_chunks > 1:
            logger.debug('Splitting the images into %i chunks.', len(slices))

        for slc_y, slc_x in slices:
            buff_shp = (len(self._images),
                        slc_y.stop-slc_y.start,
                        slc_x.stop-slc_x.start)
            integer = self._dtype.kind in 'iu'
            fill = 0 if integer else np.nan
            buffer = np.full(buff_shp, fill_value=fill, dtype=self._dtype)
            self._rejected = np.zeros(buff_shp, dtype=bool) if integer else None
            if unct:
                unct_buffer = np.full(buff_shp, fill_value=fill,
                                      dtype=self._dtype)
            else:
                unct_buffer = None

            for i, frame in enumerate(self._images):
                buffer[i] = frame.data[slc_y, slc_x]
                mask = frame.mask
                if mask is not None:
                    if integer:
                        self._rejected[i] = mask[slc_y, slc_x]
                    else:
                        buffer[i][mask[slc_y, slc_x]] = np.nan
                if unct:
                    unct_buffer[i] = frame.uncertainty[slc_y, slc_x]
                    if mask is not None and not integer:
                        unct_buffer[i][mask[slc_y, slc_x]] = np.nan

            yield buffer, unct_buffer, (slc_y, slc_x)

    def _apply_rejection(self):
        if self._rejected is not None:
            self._apply_integer_rejection()
            return
        mask = np.zeros(self._buffer[0].shape)
        if self._sigma_clip is not None:
            mask = _sigma_clip(self._buffer,
                               threshold=self._sigma_clip,
                               cen_func=self._sigma_cen_func,
                               dev_func=self._sigma_dev_func,
                               axis=0)

        if self._minmax is not None:
            _min, _max = self._minmax
            mask = np.logical_or(_minmax_clip(self._buffer, _min, _max),
                                 mask)

        mask = np.logical_or(np.isnan(self._buffer), mask)
        self._buffer[mask] = np.nan
        if self._unct_bf is not None:
            self._unct_bf[mask] = np.nan

    def _apply_integer_rejection(self):
        """Apply rejection using integer statistics and a separate mask."""
        mask = self._rejected.copy()
        if self._sigma_clip is not None:
            def statistic(func):
                if callable(func):
                    # A custom callback receives masked integer data and must
                    # respect the selected dtype in its own calculations.
                    return np.asarray(func(np.ma.array(self._buffer, mask=mask,
                                                       copy=True),
                                           axis=0), dtype=self._dtype)
                return _integer_reduce(self._buffer, mask, func)

            center = statistic(self._sigma_cen_func)
            deviation = statistic(self._sigma_dev_func)
            limits = self._sigma_clip
            low, high = (limits, limits) if np.isscalar(limits) else limits
            if low is not None:
                bound = center - self._dtype.type(low) * deviation
                self._rejected |= self._buffer < bound
            if high is not None:
                bound = center + self._dtype.type(high) * deviation
                self._rejected |= self._buffer > bound
        if self._minmax is not None:
            limits = [None if limit is None else self._dtype.type(limit)
                      for limit in self._minmax]
            self._rejected |= _minmax_clip(self._buffer, *limits)

    @_optional('IMARITH_SKIP_UNCERTAINTY', 'FRAMEDATA_DISABLE_UNCERTAINTY')
    def _combine_uncertainty(self, method, n_no_mask):
        """Estimate uncertainties for a chunk before sum normalization."""
        if self._rejected is not None:
            return self._combine_integer_uncertainty(method, n_no_mask)
        if method == 'sum':
            if self._unct_bf is None:
                logger.info('Data with no uncertainties. Using the std dev'
                            ' approximation to compute the sum uncertainty.')
                # we consider, here, that the deviation in each pixel (x, y) is
                # the error of each image in that position. So
                # unct = stddev*sqrt(n)
                unct = _funcs['std'](self._buffer, axis=0)*np.sqrt(n_no_mask)
            else:
                # direct propagate the errors in the sum
                # unct = sqrt(sigma1^2 + ) for i in sigma2^2 + ...)
                unct = _funcs['sum'](np.square(self._unct_bf), axis=0)
                unct = np.sqrt(unct)

        elif method in ('median', 'mean'):
            # uncertainty = sigma/sqrt(n)
            unct = _funcs['std'](self._buffer, axis=0)
            unct /= np.sqrt(n_no_mask)

        return unct

    def _combine_integer_uncertainty(self, method, count):
        """Propagate errors using integer accumulators and integer roots."""
        if method == 'sum' and self._unct_bf is not None:
            squared = np.square(self._unct_bf, dtype=self._dtype.type)
            variance = _integer_reduce(squared, self._rejected, 'sum')
            return _integer_sqrt(variance)
        uncertainty = _integer_reduce(self._buffer, self._rejected, 'std')
        root_count = _integer_sqrt(np.asarray(count, dtype=self._dtype))
        if method == 'sum':
            uncertainty *= root_count
        else:
            _integer_divide(uncertainty, np.maximum(root_count, 1))
        return uncertainty

    def _combine(self, method, **kwargs):
        """Combine nominal values and optionally compute uncertainties."""
        # Counts and normalization also affect nominal data when errors are off.
        n = float(len(self._buffer))
        if self._rejected is None:
            n_no_mask = n - np.sum(np.isnan(self._buffer), axis=0)
            data = _funcs[method](self._buffer, axis=0)
        else:
            n_no_mask = np.count_nonzero(~self._rejected, axis=0)
            data = _integer_reduce(self._buffer, self._rejected, method)
        unct = self._combine_uncertainty(method, n_no_mask)
        if method == 'sum' and kwargs.get('sum_normalize', True):
            if self._rejected is None:
                norm = n/n_no_mask
            else:
                norm = np.full(data.shape, n, dtype=self._dtype)
                _integer_divide(norm, np.maximum(n_no_mask, 1))
            data *= norm
            if unct is not None:
                unct *= norm
        return data, unct

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
            self._load_images(image_list)
            self._check_consistency()

            logger.info('Combining %i images with %s method.',
                        len(self._images), method)

            # temp combined data, mask and uncertainty
            data = np.zeros(self._shape, dtype=self._dtype)
            if self._dtype.kind == 'f':
                data.fill(np.nan)
            unct = None
            mask = (None if conf.FRAMEDATA_DISABLE_FLAGS
                    else np.zeros(self._shape, dtype=bool))

            for self._buffer, self._unct_bf, slc in self._chunk_yielder(method):
                # perform the masking: first with minmax, after sigma_clip
                # the clippings interfere in each other.
                self._apply_rejection()

                # combine the images and compute the uncertainty
                data[slc], chunk_uncertainty = self._combine(method, **kwargs)
                if chunk_uncertainty is not None:
                    if unct is None:
                        unct = np.empty(self._shape, dtype=self._dtype)
                    unct[slc] = chunk_uncertainty
                if mask is not None:
                    mask[slc] = (np.isnan(data[slc]) if self._rejected is None
                                 else np.all(self._rejected, axis=0))

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
      - 'median': `~numpy.nanmedian`
      - 'mean': `~numpy.nanmean`
    - The standard deviation funcion names are:
      - 'std': `~numpy.nanstd`
      - 'mad_std': `~astropy.stats.funcs.mad_std`
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
    combiner.set_minmax_clip(mm)

    merge_header = kwargs.pop('merge_header', 'no_merge')
    header_keys = kwargs.pop('merge_header_keys', None)
    combiner.set_merge_header(merge_header, header_keys)

    # Perform the combinations
    return combiner.combine(frames, method, **kwargs)
