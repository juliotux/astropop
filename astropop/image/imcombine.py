# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Stack and combine astronomical images in FrameData."""

import functools
from tempfile import mkdtemp
import numpy as np
from astropy.stats import mad_std

from ..framedata import FrameData, check_framedata
from ..py_utils import check_number
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
          Maximum memory to be used during median and mean combining.
          In bytes.
          Default: 1e9 (1GB)
        dtype: `~numpy.dtype` (optional)
          Data type to be used during the operations and the final result.
          Defualt: `~numpy.float64`
        tmp_dir: `str` (optional)
          Directory to store temporary files used in the combining. If None,
          a tmp dir will be created with `~tempfile.mkdtemp`.
          default: `None`
        use_disk_cache: `bool`
            Enable caching images to disk. This slow down the process, but
            avoid memory overflow for large number of images. If the memory
            size needed exceeds max_memory, it is enabled automatically.
            default: false
        merge_header: {'no_merge', 'first', 'only_equal', 'selected_keys'}
            Strategy for merging headers.
            default: 'no_merge'
        merge_header_keys: `list` (optional)
            Keywords for header merging if the strategy is 'selected_keys'.
            default: None
        """
        # workaround to check dtype
        if not isinstance(dtype(0), (float, np.floating)):
            raise ValueError("Only float dtypes are allowed in ImCombiner.")
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
        l, h = min_value, max_value
        # disable
        if l is None and h is None:
            self._minmax = None

        for i in (l, h):
            if not check_number(i) and i is not None:
                raise ValueError(f"{i} is not compatible with min_max "
                                 "clipping")

        # check if minimum is lower then maximum
        if l is not None and h is not None:
            l, h = sorted([l, h])

        self._minmax = (l, h)

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
        # ensure cleaning of tmp files and free memory
        for i in self._images:
            i.disable_memmap()
            i.data = None
        self._images = []
        self._shape = None
        self._unit = None

    def _load_images(self, image_list):
        """Read images to FrameData and enable memmap."""
        # clear the buffers before load images.
        self._clear()

        if len(image_list) == 0:
            raise ValueError('Image list is empty.')

        for indx, i in enumerate(image_list):
            # before combine, copy everything to FrameData
            ic = check_framedata(i, copy=True)
            ic = ic.astype(self._dtype)

            # for optimization, only enable memmap when needed.
            if self._disk_cache:
                ic.enable_memmap(filename=f'image_{indx}',
                                 cache_folder=self._tmpdir)

            self._images.append(ic)

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

    def _chunk_yielder(self, method):
        """Split the data in chuncks according to the method."""
        # sum needs uncertainties, others ignore it
        unct = False
        if method == 'sum':
            if not np.any([i.uncertainty is None for i in self._images]):
                unct = True
            else:
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
            buffer = np.full(buff_shp, fill_value=np.nan, dtype=self._dtype)
            if unct:
                unct_buffer = np.full(buff_shp, fill_value=np.nan,
                                      dtype=self._dtype)
            else:
                unct_buffer = None

            for i, frame in enumerate(self._images):
                buffer[i] = frame.data[slc_y, slc_x]
                buffer[i][frame.mask[slc_y, slc_x]] = np.nan
                if unct:
                    unct_buffer[i] = frame.uncertainty[slc_y, slc_x]
                    unct_buffer[i][frame.mask[slc_y, slc_x]] = np.nan

            yield buffer, unct_buffer, (slc_y, slc_x)

    def _apply_rejection(self):
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

    def _combine(self, method, **kwargs):
        """Process the combine and compute the uncertainty."""
        # number of masked pixels for each position
        n_masked = np.sum(np.isnan(self._buffer), axis=0)
        # number of images
        n = float(len(self._buffer))
        # number of not masked pixels for each position
        n_no_mask = n - n_masked

        if method == 'sum':
            data = _funcs['sum'](self._buffer, axis=0)
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

            if kwargs.get('sum_normalize', True):
                norm = n/n_no_mask
                data *= norm
                unct *= norm

        elif method in ('median', 'mean'):
            data = _funcs[method](self._buffer, axis=0)
            # uncertainty = sigma/sqrt(n)
            unct = _funcs['std'](self._buffer, axis=0)
            unct /= np.sqrt(n_no_mask)

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

        # first of all, load the images to FrameData and check the consistency
        self._load_images(image_list)
        self._check_consistency()

        logger.info('Combining %i images with %s method.',
                    len(self._images), method)

        # temp combined data, mask and uncertainty
        data = np.zeros(self._shape, dtype=self._dtype)
        data.fill(np.nan)
        unct = np.zeros(self._shape, dtype=self._dtype)
        mask = np.zeros(self._shape, dtype=bool)

        for self._buffer, self._unct_bf, slc in self._chunk_yielder(method):
            # perform the masking: first with minmax, after sigma_clip
            # the clippings interfere in each other.
            self._apply_rejection()

            # combine the images and compute the uncertainty
            data[slc], unct[slc] = self._combine(method, **kwargs)
            mask[slc] = np.isnan(data[slc])

        n = len(self._images)
        combined = FrameData(data, unit=self._unit, uncertainty=unct,
                             mask=mask)
        combined.meta = merge_header(*[i.header for i in self._images],
                                     method=self._header_strategy,
                                     selected_keys=self._header_merge_keys)
        combined.meta['HIERARCH astropop imcombine nimages'] = n
        combined.meta['HIERARCH astropop imcombine method'] = method

        # after, clear all buffers
        self._clear()
        return combined


def imcombine(frames, method='median', memory_limit=1e9, **kwargs):
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
        The maximum memory limit (in bytes) to be used in the combining.
        If the data exceeds the maximum memory limit, it will be slipted in
        chunks for the rejection and combining processes.
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
    kargs = {'dtype': kwargs.pop('dtype', np.float64),
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
