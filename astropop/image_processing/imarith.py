# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Handle the IRAF's imarith and imcombine functions."""

import numpy as np
from astropy.units.core import UnitConversionError
from astropy.stats import sigma_clip as sc

from ..framedata import FrameData, check_framedata
from ..math.physical import QFloat, convert_to_qfloat, UnitsError
from ..py_utils import check_iterable, check_number
from ..logger import logger, log_to_list

# if bottleneck is available, use it
try:
    import bottleneck as bn
except ImportError:
    HAS_BOTTLENECK = False
else:
    HAS_BOTTLENECK = True

_median = bn.nanmedian if HAS_BOTTLENECK else np.nanmedian
_mean = bn.nanmean if HAS_BOTTLENECK else np.nanmean
_std = bn.nanstd if HAS_BOTTLENECK else np.nanstd
_sum = bn.nansum if HAS_BOTTLENECK else np.nansum


__all__ = ['imarith', 'imcombine', 'ImCombiner']


# imatrith related functions


_arith_funcs = {'+': np.add,
                '-': np.subtract,
                '/': np.true_divide,
                '//': np.floor_divide,
                '*': np.multiply,
                '**': np.power,
                '%': np.remainder}


def _qf_or_framedata(data, alternative=convert_to_qfloat):
    """Check if the data is QFloat or FrameData. Else, convert it."""
    if isinstance(data, (QFloat, FrameData)):
        return data
    return alternative(data)


def _arith_mask(operand1, operand2):
    """Handle the arithmatics of the masks."""
    def _extract(operand):
        if hasattr(operand, 'mask'):
            return operand.mask
        return False

    mask1 = _extract(operand1)
    mask2 = _extract(operand2)

    old_n = np.count_nonzero(mask1)
    nmask = np.logical_or(mask1, mask2)
    new_n = np.count_nonzero(nmask)
    logger.debug('Updating mask in math operation. '
                 'From %i to %i masked elements.', old_n, new_n)
    return nmask


def _arith(operand1, operand2, operation):
    """Perform the math operation itself using QFloats."""
    qf1 = convert_to_qfloat(operand1)
    qf2 = convert_to_qfloat(operand2)

    res = _arith_funcs[operation](qf1, qf2)
    return res.nominal, res.std_dev


def _join_headers(operand1, operand2, operation):  # noqa
    """Join the headers to result."""
    # TODO: Think if this is the best behavior
    return operand1.header.copy()


def imarith(operand1, operand2, operation, inplace=False,
            join_masks=False):
    """Perform arithmetic operations using `~astropop.framedata.FrameData`.

    Notes
    -----
    * Keeps the header of the first image.

    * If ``operand1`` is not a `~astropop.framedata.FrameData` instance,
      inplies in ``inplace=False``, and a new `~astropop.framedata.FrameData`
      instance will be created.

    * Supported operations:
        - ``+`` : add. Example: 1+1=2
        - ``-`` : subtract. Example: 2-1=1
        - ``*`` : scalar product. Example: 2*3=6
        - ``/`` : true division. Example: 3/2=1.5
        - ``**`` : power. Example: 3**2=9
        - ``%`` : modulus. Example: 7%2=1
        - ``//`` : floor division. Example: 7//2=3

    Parameters
    ----------
    operand1, operand2: `~astropop.framedata.FrameData` compatible
        Values to perform the operation. `~astropop.math.physical.QFloat`
        `~astropy.units.Quantity`, numerical values and
        `~astropy.nddata.CCDData` are also suported.
    operation: {``+``, ``-``, ``*``, ``/``, ``**``, ``%``, ``//``}
        Math operation.
    inplace: bool, optional
        If True, the operations will be performed inplace in the operand 1.
    join_masks: bool, optional
        Join masks in the end of the operation.

    Returns
    -------
    `~astropop.framedata.FrameData`:
        new `FrameData` instance if not ``inplace``, else the ``operand1``
        `~astropop.framedata.FrameData` instance.
    """
    # TODO: handle WCS
    if operation not in _arith_funcs.keys():
        raise ValueError(f"Operation {operation} not supported.")

    if isinstance(operand1, FrameData) and inplace:
        ccd = operand1
    else:
        ccd = FrameData(None)

    operand1 = _qf_or_framedata(operand1)
    operand2 = _qf_or_framedata(operand2)

    # Add the operation entry to the ccd history.
    lh = log_to_list(logger, ccd.history)
    logger.debug('Operation %s between %s and %s',
                 operation, operand1, operand2)

    # Perform data, mask and uncertainty operations
    try:
        ccd.data, ccd.uncertainty = _arith(operand1, operand2, operation)
    except UnitConversionError:
        raise UnitsError(f'Units {operand1.unit} and {operand2.unit} are'
                         f' incompatible for {operation} operation.')

    if join_masks:
        ccd.mask = _arith_mask(operand1, operand2)
    else:
        ccd.mask = False

    ccd.meta = _join_headers(operand1, operand2, operation)

    logger.removeHandler(lh)

    return ccd


###############################################################################
# imcombine related functions
###############################################################################


def _sigma_clip(data, threshold=3, cen_func=np.nanmedian, dev_func=np.nanstd,
                axis=None):
    """Create a mask of the sigma clipped pixels.ccdclip.

    It uses the `~astropy.stats.sigma_clip` to perform sigmaclipping on the
    data. This function will not change the array, instead, just output a
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
    def_func: callable or {'std'} (optional)
        Function to compute the base deviation value used for sigma clipping.
        Default: 'std'
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
    elif check_iterable(threshold):
        slow, shigh = threshold
    else:
        raise TypeError(f'Sigma clipping threshold {threshold} not'
                        ' recognized.')

    mask = sc(data, sigma_lower=slow, sigma_upper=shigh, maxiters=1,
              axis=axis, cenfunc=cen_func, stdfunc=dev_func, copy=True,
              masked=True).mask

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

    return mask


class ImCombiner:
    """Process the combining operation of images, like the IRAF imcombine."""

    _sigma_clip = None
    _sigma_cen_func = None
    _sigma_dev_func = None
    _minmax = None
    _max_memory = 1e8
    _buffer = None
    _images = None

    _methods = {'median', 'mean', 'sum'}

    def __init__(self, max_memory=1e9):
        """Combine images using various algorithms.

        Parameters
        ----------
        - max_memory: int (optional)
          Maximum memory to be used during median and mean combining.
          In bytes.
          Default: 1e9
        """
        self._max_memory = max_memory
        # initialize empty image list
        self._images = []

    def set_sigma_clip(self, sigma_limits=None,
                       center_func='median', dev_func='std'):
        """Enable sigma clipping during the combine.

        Parameters
        ----------
        - sigma_limits: `float`, `tuple` or `None` (optional)
          Set the low and high thresholds for sigma clipping. A number is
          applyed to both low and high limits. A tuple will be considered
          (low, high) limits. `None` disable the clipping.
          Default: `None`
        - center_func: callable or {'median', 'mean'} (optional)
          Function to compute de central tendency of the data.
          Default: 'median'
        - dev_func: callable or {'std', 'mad_std'} (optional)
          Function to compute the deviation sigma for clipping.
          Defautl: 'std'
        """
        if sigma_limits is None:
            # None simply disables sigma clipping
            self._sigma_clip = None
            self._sigma_cen_func = None
            self._sigma_dev_func = None
            return

        if check_iterable(sigma_limits):
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
        - min_value: `float` or `None` (optional)
          Minimum threshold of the clipping. `None` disables minimum masking.
          Default: `None`
        - max_value: `float` or `None` (optional)
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
            l, h = (l, h) if l < h else (h, l)

        self._minmax = (l, h)

    def _clear(self):
        """Clear buffer and images."""
        self._buffer = None
        # ensure cleaning of tmp files and free memory
        for i in self._images:
            i.disable_memmap()
            i.data = None
        self._images = []

    def _load_images(self, image_list):
        """Read images to FrameData and enable memmap."""
        # clear the buffers before load images.
        self._clear()

        if len(image_list) == 0:
            raise ValueError('Image list is empty.')

        is_not_framedata = False
        for i in image_list:
            # before combine, copy everything to memmaped FrameData
            if not isinstance(i, FrameData) and not is_not_framedata:
                logger.warning('The images to combine are not FrameData. '
                               'Some features may be disabled.')
                is_not_framedata = True
            ic = check_framedata(i, copy=True)
            ic.enable_memmap()
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

    def _chunk_yielder(self, method):
        """Split the data in chuncks according to the method."""
        n = len(self._images)
        shape = self._images[0].shape

        imsize = self._images[0].data.nbytes
        imsize += self._images[0].mask.nbytes
        # uncertainty is ignored

        tot_size = n*imsize
        # adjust memory usage for numpy and bottleneck
        if method == 'median':
            tot_size *= 4.5
        else:
            tot_size *= 3

        n_chunks = np.ceil(tot_size/self._max_memory)

        # compute x and y steps
        xstep = max(1, int(shape[0]/n_chunks))
        if shape[0] >= n_chunks:
            ystep = shape[1]
        else:
            ystep = max(1, int(shape[1]/int(n_chunks/shape[0])))

        n_chunks = np.ceil(shape[0]/xstep)*np.ceil(shape[1]/ystep)
        if n_chunks == 1:
            result = [np.ma.MaskedArray(i.data, i.mask) for i in self._images]
            yield result, (slice(0, shape[0]), slice(0, shape[1]))
        else:
            logger.debug('Splitting the images into %i chunks.', n_chunks)
            # return the sliced data and the slice
            for x in range(0, shape[0], xstep):
                for y in range(0, shape[1], ystep):
                    slc_x = slice(x, min(x+xstep, shape[0]))
                    slc_y = slice(y, min(y+ystep, shape[1]))
                    l = [np.ma.MaskedArray(i.data[slc_x, slc_y],
                                        mask=i.mask[slc_x, slc_y])
                        for i in self._images]
                    yield l, (slc_x, slc_y)

    def combine(self, image_list, method, weights=None, scale=None, **kwargs):
        """Perform the image combining.

        Parameters
        ----------
        - image_list: `list` or `tuple`
          List containing the images to be combined. The values in the list
          must be all of the same type and `~astropop.framedata.FrameData`
          supported.
        - method: {'mean', 'median', 'sum'}
          Combining method.

        Returns
        -------
        - combined: `~astropop.framedata.FrameData`
          The combined image.

        Notes
        -----
        - For now, it don't consider WCS, so it perform plain between the
          images, whitout registering.
        - Clipping parameters are set using class functions.
        - If the images exceed the maximum memory allowed, they are splited
          to perform the median and mean combine.
        - Masked elements are skiped.
        """
        # first of all, load the images to FrameData and check the consistency
        self._load_images(image_list)
        self._check_consistency()

        if method not in self._methods:
            raise ValueError(f'{method} is not a valid combining method.')

        # TODO: check consistency with weitghts and scale

        # TODO: perform the masking

        # TODO: split the chuncks and combine
        for chunk, slc in self._chunk_yielder(method):
            logger.debug(slc)

        # after, clear all buffers
        self._clear()


def imcombine(frames, method='median', sigma_clip=None,
              sigma_cen_func='median', sigma_dev_func='std',
              minmax_clip=None, memory_limit=1e9):
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
    memory_limit: int (optional)
        The maximum memory limit (in bytes) to be used in the combining.
        If the data exceeds the maximum memory limit, it will be slipted in
        chunks for the rejection and combining processes.

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
    # TODO: this is just a wrapper for the ImCombiner
    raise NotImplementedError
