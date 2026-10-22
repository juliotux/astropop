# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Compute shifts and translate astronomical images."""

from __future__ import annotations

import typing as _t
import numpy.typing as _npt
from .._typing import ImageLike as _ImageLike
from .._typing import Window as _Window
from skimage.transform import AffineTransform as _AffineTransform

import abc
from functools import partial
from skimage import transform
import numpy as np

from .processing import trim_image
from ..logger import logger
from ..framedata import check_framedata, FrameData, PixelMaskFlags


__all__ = ['CrossCorrelationRegister', 'AsterismRegister',
           'register_framedata_list', 'compute_shift_list']


def _algorithm_check(
    algorithm: _t.Literal['cross-correlation', 'asterism-matching'],
    kwargs: dict[str, _t.Any],
) -> _BaseRegister:
    """Check the algorithms and get the register."""
    if algorithm == 'cross-correlation':
        return CrossCorrelationRegister(**kwargs)
    if algorithm == 'asterism-matching':
        return AsterismRegister(**kwargs)
    raise ValueError(f'Algorithm {algorithm} unknown.')


def _check_compatible_list(frame_list: _t.Sequence[FrameData]) -> None:
    """Check if the list if it is compatible with the register."""
    for i in frame_list:
        if not isinstance(i, FrameData):
            raise TypeError('Only a list of FrameData instances is allowed.')
        if i.shape != frame_list[0].shape:
            raise ValueError('Images with incompatible shapes '
                             f'{frame_list[0].shape} and {i.shape}.'
                             'Only frames with same shape allowed.')


def _get_clip_slices(
    shifts: _npt.ArrayLike,
    imshape: tuple[int, int],
) -> tuple[slice, slice]:
    """Get the slices of the clip."""
    # negative shifts means the image is translated up and right, so they
    # define the start.
    # positive shifts translates down and left, so they define the stop.
    xmin, xmax = np.nanmin(shifts[:, 0]), np.nanmax(shifts[:, 0])
    ymin, ymax = np.nanmin(shifts[:, 1]), np.nanmax(shifts[:, 1])

    xstart = int(np.ceil(-xmin)) if xmin < 0 else 0
    xstop = int(np.floor(-xmax)) if xmax > 0 else imshape[0]

    ystart = int(np.ceil(-ymin)) if ymin < 0 else 0
    ystop = int(np.floor(-ymax)) if ymax > 0 else imshape[1]

    return slice(xstart, xstop), slice(ystart, ystop)


_keywords = {
    'method': 'HIERARCH astropop registration',
    'shift_x': 'HIERARCH astropop registration_shift_x',
    'shift_y': 'HIERARCH astropop registration_shift_y',
    'rotation': 'HIERARCH astropop registration_rot',
}


class _BaseRegister(abc.ABC):
    """Base class for Registers."""

    _name = None

    def __init__(self, window: _Window=None) -> None:
        """Validate optional row and column slices with unit steps and store the window."""
        if window is not None:
            if (not isinstance(window, tuple) or len(window) != 2
                    or not all(isinstance(i, slice) for i in window)):
                raise ValueError('window must be a tuple of two slices '
                                 '(rows, columns).')
            if any(i.step not in (None, 1) for i in window):
                raise ValueError('window slices must have unit steps.')
        self.window = window

    def _window_slices(
        self,
        image1: _ImageLike,
        image2: _ImageLike,
    ) -> tuple[slice, slice]:
        """Validate and normalize the window against both image shapes."""
        if image1.ndim != 2 or image1.shape != image2.shape:
            raise ValueError('Window registration requires matching 2D shapes.')
        slices = []
        for section, size in zip(self.window, image1.shape):
            start, stop, _ = section.indices(size)
            if start >= stop:
                raise ValueError('window must select a non-empty image region.')
            slices.append(slice(start, stop))
        return tuple(slices)

    @staticmethod
    def _apply_transform_image(
        image: _npt.NDArray,
        tform: _AffineTransform,
        cval: float=0,
        order: int=3,
    ) -> _npt.NDArray:
        """Apply the transform to an image."""
        # ensure correct data types
        image = np.array(image)
        cval = float(cval)

        # apply the transform
        return transform.warp(image, tform, mode='constant', cval=cval,
                              preserve_range=True, order=order)

    @abc.abstractmethod
    def _compute_transform(
        self,
        image1: _ImageLike,
        image2: _ImageLike,
        mask1: _ImageLike | None=None,
        mask2: _ImageLike | None=None,
    ) -> _AffineTransform:
        """Compute the transform to register image2 to image1."""

    def compute_transform(
        self,
        image1: _ImageLike,
        image2: _ImageLike,
        mask1: _ImageLike | None=None,
        mask2: _ImageLike | None=None,
    ) -> _AffineTransform:
        """Compute the transform to register image2 to image1.

        Parameters
        ----------
        image1
            Reference image for registration.
        image2
            Moving image for registration.
        mask1, mask2
            Masks for images 1 and 2.
            Default: `None`

        Returns
        -------
        tform
            Transform in full-image coordinates.

        Notes
        -----
        Only pixels inside the configured ``window`` are used to estimate
        the transform.
        """
        if self.window is None:
            return self._compute_transform(image1, image2, mask1, mask2)

        image1, image2 = np.asarray(image1), np.asarray(image2)
        window = self._window_slices(image1, image2)
        mask1 = None if mask1 is None else np.asarray(mask1)[window]
        mask2 = None if mask2 is None else np.asarray(mask2)[window]
        tform = self._compute_transform(image1[window], image2[window],
                                        mask1, mask2)
        # The algorithm returns a mapping from reference to moving pixels.
        # Translate its local window coordinates back to full-image coordinates.
        origin = transform.AffineTransform(
            translation=(window[1].start, window[0].start)).params
        return transform.AffineTransform(
            matrix=origin @ tform.params @ np.linalg.inv(origin))

    def register_image(
        self,
        image1: _ImageLike,
        image2: _ImageLike,
        mask1: _ImageLike | None=None,
        mask2: _ImageLike | None=None,
        cval: float | _t.Literal['median', 'mean']='median',
    ) -> tuple[_npt.NDArray, _npt.NDArray[np.bool_], _AffineTransform]:
        """Align and transform an image2 to match image1.

        Parameters
        ----------
        image1
            Reference image for registration.
        image2
            Moving image for registration.
        mask1, mask2
            Masks for images 1 and 2.
            Default: `None`
        cval
            Fill value for transformed pixels from outside the image. If
            'median' or 'mean', these statistics will be computed as cval.
            Default: 'median'

        Returns
        -------
        reg_image
            Registered image according the transform computed by the class.
        mask
            Mask for the registered image.
        tform
            Sampling transform from reference to moving image coordinates.
        """
        if self.window is not None:
            self._window_slices(np.asarray(image1), np.asarray(image2))

        # equal images are just returned
        if np.all(image1 == image2):
            logger.info('Images are equal, skipping registering.')
            return (image1, np.zeros_like(image1, dtype=bool),
                    transform.AffineTransform(translation=(0, 0)))

        tform = self.compute_transform(image1, image2, mask1, mask2)
        if mask2 is None:
            mask2 = np.zeros_like(image2)

        # TODO: consider the mask here
        if cval == 'median':
            cval = np.nanmedian(image2)
        if cval == 'mean':
            cval = np.nanmean(image2)

        reg_image = self._apply_transform_image(image2, tform, cval=cval)
        logger.info('Filling registered image with cval=%.2f', cval)
        logger.info('Registering image with: '
                    'translation=%s, rotation=%.2f°',
                    tform.translation, np.rad2deg(tform.rotation))

        # use float to get partial covered pixels and mask them
        mask = self._apply_transform_image(mask2.astype('f8'), tform, cval=1,
                                           order=0)
        mask = mask > 0
        return reg_image, mask, tform

    def register_framedata(
        self,
        frame1: FrameData,
        frame2: FrameData,
        cval: float | _t.Literal['median', 'mean']='median',
        inplace: bool=False,
    ) -> FrameData:
        """Align and transform a frame2 to match frame1.

        Parameters
        ----------
        frame1
            Reference image for registration.
        frame2
            Moving image for registration.
        cval
            Fill value for transformed pixels from outside the image. If
            'median' or 'mean', these statistics will be computed as cval.
            Default: 'median'
        inplace
            Perform the operation in original frame2 instance. The original
            data will be changed.

        Returns
        -------
        reg_frame
            Registered frame.
        """
        frame1 = check_framedata(frame1)
        frame2 = check_framedata(frame2)

        im1 = np.array(frame1.data)
        im2 = np.array(frame2.data)
        msk1 = frame1.mask
        msk2 = frame2.mask

        data, mask, tform = self.register_image(im1, im2, msk1, msk2,
                                                cval=cval)

        if inplace:
            reg_frame = frame2
        else:
            # Copy the frame to mantain the memmap caching behavior
            reg_frame = frame2.copy()

        reg_frame.data = data
        flags = frame2.flags
        if flags is not None:
            flags = self._apply_transform_image(flags, tform, cval=0, order=0)
            reg_frame.flags = flags
        reg_frame.add_flags(PixelMaskFlags.OUT_OF_BOUNDS |
                            PixelMaskFlags.MASKED,
                            mask)

        unct = frame2.get_uncertainty()
        if unct is not None:
            unct = self._apply_transform_image(unct, tform, cval=np.nan,
                                               order=1)
        reg_frame.uncertainty = unct

        sx, sy = tform.translation
        theta = np.rad2deg(tform.rotation)
        reg_frame.meta[_keywords['method']] = self._name
        reg_frame.meta[_keywords['shift_x']] = sx
        reg_frame.meta[_keywords['shift_y']] = sy
        reg_frame.meta[_keywords['rotation']] = theta

        if reg_frame.wcs is not None:
            logger.warn('WCS in frame2 is not None. Due to the transform it'
                        ' will be erased.')
            reg_frame.wcs = None

        return reg_frame


class CrossCorrelationRegister(_BaseRegister):
    """Register images using `~skimage.registration.phase_cross_correlation`.

    It uses cross-correlation to find a translation-only transform between
    two images. It obtains an initial estimate of the cross-correlation
    peak by an FFT and then refines the shift estimation by upsampling
    the DFT only in a small neighborhood of that estimate by means of a
    matrix-multiply DFT[1]_.

    Parameters
    ----------
    window
        Region used to estimate the transform, in NumPy (rows, columns) order.
        For example, ``(slice(100, 300), slice(200, 400))``. The same region
        is selected in both images; only its pixels are used for registration.
        Slices follow NumPy bounds and must have unit steps and select a
        non-empty region. The transform is applied to the full image.
        Default: `None` (use the full image).
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel. Default is 1 (no upsampling).
        Not used if any of ``reference_mask`` or ``moving_mask`` is not None.
    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data. "real" means
        data will be FFT'd to compute the correlation, while "fourier"
        data will bypass FFT of input data. Case insensitive. Not
        used if any of ``reference_mask`` or ``moving_mask`` is not
        None.
    disambiguate : bool
        The shift returned by this function is only accurate *modulo* the
        image shape, due to the periodic nature of the Fourier transform. If
        this parameter is set to ``True``, the *real* space cross-correlation
        is computed for each possible shift, and the shift with the highest
        cross-correlation within the overlapping area is returned.
    return_error : bool, {"always"}, optional
        Returns error and phase difference if "always" is given. If False, or
        either ``reference_mask`` or ``moving_mask`` are given, only the shift
        is returned.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower ``overlap_ratio`` leads to smaller
        maximum translation, while a higher ``overlap_ratio`` leads to greater
        robustness against spurious matches due to small overlap between
        masked images. Used only if one of ``reference_mask`` or
        ``moving_mask`` is not None.

    Notes
    -----
    - Due to a bug in the `~skimage.registration.phase_cross_correlation`
      normalization is automatically disabled.
    - On older scikit-image versions, ``return_error`` is set to ``'always'``
      for compatibility with the expected return format.
    - Input masks passed to the registration methods are currently ignored
      during transform estimation. Use ``window`` to select a region.
    - A window requires real-space input (``space='real'``).

    References
    ----------
    .. [1] :doi:`10.1364/OL.33.000156`

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> reference = np.zeros((24, 24))
    >>> reference[8:12, 10:14] = [[1, 2, 1, 0], [2, 5, 2, 1],
    ...                         [1, 3, 2, 0], [0, 1, 0, 0]]

    >>> # NumPy roll takes (dy, dx); reported registration shifts use (dx, dy).
    >>> moving = np.roll(reference, (2, -3), axis=(0, 1))

    >>> from astropop.image.register import CrossCorrelationRegister
    >>> register = CrossCorrelationRegister(upsample_factor=10)
    >>> transform = register.compute_transform(reference, moving)
    >>> print(np.round(transform.translation, 2).tolist())
    [-3.0, 2.0]
    """

    _name = 'cross-correlation'

    def __init__(self, window: _Window=None, **kwargs: _t.Any) -> None:
        """Configure phase correlation and require real-space data for a window."""
        super().__init__(window=window)
        if window is not None and kwargs.get('space', 'real').lower() != 'real':
            raise ValueError('window requires real-space image data.')
        import skimage
        from skimage.registration import phase_cross_correlation
        if skimage.__version__ >= '0.19':
            kwargs['normalization'] = None
        if skimage.__version__ <= '0.21':
            kwargs['return_error'] = 'always'
        self._pcc = partial(phase_cross_correlation, **kwargs)

    def _compute_transform(
        self,
        image1: _ImageLike,
        image2: _ImageLike,
        mask1: _ImageLike | None=None,
        mask2: _ImageLike | None=None,
    ) -> _AffineTransform:
        """Estimate the translation from phase correlation, ignoring supplied masks.

        Convert the returned row/column displacement into an affine x/y transform
        with the sign convention expected by the registration application step.
        """
        if mask1 is not None or mask2 is not None:
            logger.debug("Masks are ignored in CrossCorrelationRegister.")
        # Masks are ignored by default
        dy, dx = self._pcc(image1, image2)[0]
        return transform.AffineTransform(translation=(-dx, -dy))


class AsterismRegister(_BaseRegister):
    """Register images using asterism matching. Based on astroalign [1]_.

    Match triangles of detected sources to estimate translation, rotation,
    and scale between two images. Sources are detected with an astropop
    photometry detector and sorted by brightness before matching. The
    brightest sources, up to ``max_control_points``, are passed to astroalign.
    Background estimation and detection use only the configured window.
    Input masks are currently ignored during transform estimation.

    Parameters
    ----------
    window
        Region used to estimate the transform, in NumPy (rows, columns) order.
        For example, ``(slice(100, 300), slice(200, 400))``. The same region
        is selected in both images; only its pixels are used for registration.
        Slices follow NumPy bounds and must have unit steps and select a
        non-empty region. The transform is applied to the full image.
        Default: `None` (use the full image).
    max_control_points
        Maximum control points (stars) used in asterism matching.
        Default: 50
    detection_threshold
        Minimum SNR detection threshold.
        Default: 5
    detection_function
        Detection function to use.
        Default: 'segfind'
    **detection_kwargs :
        Keyword arguments to pass to the detection function.

    Raises
    ------
    ImportError
        If astroalign is not installed.

    References
    ----------
    .. [1] :doi:`10.1016/j.ascom.2020.100384`

    Examples
    --------
    >>> from astropop.image.register import AsterismRegister

    >>> # Configure the estimator without detecting sources or transforming pixels.
    >>> register = AsterismRegister(max_control_points=20, detection_threshold=5)
    >>> print(type(register).__name__)
    AsterismRegister

    For a complete source-matching example, see the image registration module guide.
    """

    _name = 'asterism-matching'

    def __init__(
        self,
        max_control_points: int=50,
        detection_threshold: float=5,
        detection_function: _t.Literal['segfind', 'daofind', 'starfind'] | _t.Callable[..., _t.Any]='segfind',
        window: _Window=None,
        **detection_kwargs: _t.Any,
    ) -> None:
        """Configure source detection, control-point limits and the registration window."""
        super().__init__(window=window)
        try:
            import astroalign
        except ImportError:
            raise ImportError('AsterismRegister requires astroalign tools.')

        from ..photometry import segfind, starfind, daofind, background

        funcs = {'segfind': segfind, 'starfind': starfind, 'daofind': daofind}
        self._aa = astroalign
        self._sf = partial(funcs[detection_function], **detection_kwargs)
        self._bkg = background
        self._max_cntl_pts = max_control_points
        self._threshold = detection_threshold

    def _compute_transform(
        self,
        image1: _ImageLike,
        image2: _ImageLike,
        mask1: _ImageLike | None=None,
        mask2: _ImageLike | None=None,
    ) -> _AffineTransform:
        """Detect bright sources and match asterisms to estimate a similarity transform.

        Estimate background levels independently for the two images and retain the
        brightest control points. Supplied pixel masks are currently ignored.
        """
        if mask1 is not None or mask2 is not None:
            logger.debug("Masks are ignored in AsterismRegister.")

        # use our starfind to work with only good sources
        bkg, rms = self._bkg(image1, global_bkg=True)
        sources1 = self._sf(image1, self._threshold, bkg, rms)
        bkg, rms = self._bkg(image2, global_bkg=True)
        sources2 = self._sf(image2, self._threshold, bkg, rms)
        sources1.sort('flux', reverse=True)
        sources2.sort('flux', reverse=True)
        logger.debug('Asterism matching: %d sources in image1, %d in image2',
                     len(sources1), len(sources2))
        sources1 = np.array(list(zip(sources1['x'], sources1['y'])))
        sources2 = np.array(list(zip(sources2['x'], sources2['y'])))

        tform, ctl_pts = self._aa.find_transform(sources1, sources2,
                                                 self._max_cntl_pts)

        logger.debug("Asterism matching performed with sources at: "
                     "image1: %s; image2 %s",
                     ctl_pts[0].tolist(), ctl_pts[1].tolist())

        return tform


def compute_shift_list(
    frame_list: _t.Sequence[FrameData],
    algorithm: _t.Literal['cross-correlation', 'asterism-matching']='cross-correlation',
    ref_image: int=0,
    skip_failure: bool=False,
    window: _Window=None,
    **kwargs: _t.Any,
) -> list[tuple[float, float]]:
    """Compute the shift between a list of frames.

    Parameters
    ----------
    frame_list
        A list containing `~astropop.framedata.FrameData` images to be
        registered. All images must have the same shape.
    algorithm
        The algorithm to compute the `~skimage.transform.AffineTransform`
        between the images.
        'cross-correlation' will compute the transform
        using `~skimage.registration.phase_cross_correlation` method.
        'asterism-matching' will use `~astroalign` to match asterisms of 3
        detected stars in the field and compute the transform.
        Default: 'cross-correlation'
    ref_image
        Reference image index to compute the registration.
        Default: 0
    skip_failure
        If True, failed images retain their list positions and their
        shifts will be set to nan.
        Default: False
    window
        Region used to estimate the transform, in NumPy (rows, columns) order.
        For example, ``(slice(100, 300), slice(200, 400))``. The same region
        is selected in both images; only its pixels are used for registration.
        Slices follow NumPy bounds and must have unit steps and select a
        non-empty region. Shifts are reported in full-image coordinates.
        Default: `None` (use the full image).
    **kwargs
        keyword arguments to be passed to `CrossCorrelationRegister` or
        `AsterismRegister` during instance creation. See the parameters in
        each class documentation.

    Returns
    -------
    shifts
        One ``[dx, dy]`` translation in pixels per input frame, in input order.
        The reference entry is ``[0, 0]``. Transforms map reference coordinates
        to moving-image coordinates, so alignment moves features in the
        opposite direction. Rotation and scale are not returned.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> reference = np.zeros((24, 24))
    >>> reference[8:12, 10:14] = [[1, 2, 1, 0], [2, 5, 2, 1],
    ...                         [1, 3, 2, 0], [0, 1, 0, 0]]

    >>> # NumPy roll takes (dy, dx); reported registration shifts use (dx, dy).
    >>> moving = np.roll(reference, (2, -3), axis=(0, 1))

    >>> from astropop.image.register import compute_shift_list
    >>> shifts = compute_shift_list([FrameData(reference), FrameData(moving)])
    >>> print(np.round(shifts, 2).tolist())
    [[0.0, 0.0], [-3.0, 2.0]]
    """
    reg = _algorithm_check(algorithm, dict(kwargs, window=window))
    _check_compatible_list(frame_list)

    n = len(frame_list)

    ref = frame_list[ref_image]
    ref_im = np.array(ref.data)
    ref_mk = ref.mask if ref.mask is None else np.array(ref.mask)

    shift_list = [None]*n
    for i in range(n):
        logger.info('Computing shift of image %i from %i', i+1, n)
        if i == ref_image:
            shift_list[i] = [0, 0]
            continue

        mov = frame_list[i]
        mov_im = np.array(mov.data)
        mov_mk = mov.mask if mov.mask is None else np.array(mov.mask)
        try:
            tform = reg.compute_transform(ref_im, mov_im, ref_mk, mov_mk)
            shift_list[i] = list(tform.translation)
        except Exception as e:
            logger.warning('Failed to compute shift of image %i: %s', i+1, e)
            if skip_failure:
                shift_list[i] = [np.nan, np.nan]
            else:
                raise e

    return shift_list


def register_framedata_list(
    frame_list: list[FrameData],
    algorithm: _t.Literal['cross-correlation', 'asterism-matching']='cross-correlation',
    ref_image: int=0,
    clip_output: bool=False,
    cval: float | _t.Literal['median', 'mean']='median',
    inplace: bool=False,
    skip_failure: bool=False,
    window: _Window=None,
    **kwargs: _t.Any,
) -> list[FrameData]:
    """Perform registration in a framedata list.

    Parameters
    ----------
    frame_list
        A list containing `~astropop.framedata.FrameData` images to be
        registered. All images must have the same shape.
    algorithm
        The algorithm to compute the `~skimage.transform.AffineTransform`
        between the images.
        'cross-correlation' will compute the transform
        using `~skimage.registration.phase_cross_correlation` method.
        'asterism-matching' will use `~astroalign` to match asterisms of 3
        detected stars in the field and compute the transform.
        Default: 'cross-correlation'
    ref_image
        Reference image index to compute the registration.
        Default: 0
    clip_output
        If True, trim the outputs using their measured translations.
        Rotation and scale are not included in the clipping calculation.
    cval
        Fill value for the empty pixels in the transformed image. If 'mean' or
        'median', the correspondent values will be computed from the image.
        Default: 'median'
    skip_failure
        If True, failed images retain their list positions. Their data
        will be filled with cval and all pixels will be masked.
        If False, the error will be raised.
        Default: False
    inplace
        Perform the operation inplace, modifying the original FrameData
        container. If `False`, a new container will be created.
        Default: `False`
    window
        Region used to estimate the transform, in NumPy (rows, columns) order.
        For example, ``(slice(100, 300), slice(200, 400))``. The same region
        is selected in both images; only its pixels are used for registration.
        Slices follow NumPy bounds and must have unit steps and select a
        non-empty region. The transform is applied to the full image.
        Default: `None` (use the full image).
    **kwargs
        keyword arguments to be passed to `CrossCorrelationRegister` or
        `AsterismRegister` during instance creation. See the parameters in
        each class documentation.

    Returns
    -------
    registered
        Registered frames in input order, including the reference and any
        failed frames retained by ``skip_failure=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> reference = np.zeros((24, 24))
    >>> reference[8:12, 10:14] = [[1, 2, 1, 0], [2, 5, 2, 1],
    ...                         [1, 3, 2, 0], [0, 1, 0, 0]]

    >>> # NumPy roll takes (dy, dx); reported registration shifts use (dx, dy).
    >>> moving = np.roll(reference, (2, -3), axis=(0, 1))

    >>> from astropop.image.register import register_framedata_list
    >>> aligned = register_framedata_list(
    ...     [FrameData(reference), FrameData(moving)], cval=0,
    ...     window=(slice(4, 18), slice(4, 18)))

    >>> # The window estimates the transform; the full image is returned.
    >>> print(aligned[1].shape)
    (24, 24)
    >>> print(float(np.max(np.abs(aligned[1].data - reference))))
    0.0
    """
    reg = _algorithm_check(algorithm, dict(kwargs, window=window))
    _check_compatible_list(frame_list)

    n = len(frame_list)
    reg_list = [None]*n
    for i in range(n):
        logger.info('Registering image %i from %i', i+1, n)
        try:
            reg_list[i] = reg.register_framedata(frame_list[ref_image],
                                                 frame_list[i],
                                                 cval=cval, inplace=inplace)
        except Exception as e:
            if not skip_failure:
                raise e
            logger.warning('Failed to register image %i: %s', i+1, e)
            reg_list[i] = check_framedata(frame_list[i], copy=not inplace)
            if cval == 'median':
                icval = np.median(reg_list[i].data)
            elif cval == 'mean':
                icval = np.mean(reg_list[i].data)
            else:
                icval = cval
            reg_list[i].data[:] = icval
            reg_list[i].add_flags(PixelMaskFlags.MASKED |
                                  PixelMaskFlags.OUT_OF_BOUNDS,
                                  np.ones_like(reg_list[i].data, dtype=bool))
            reg_list[i].meta[_keywords['method']] = 'failed'
            reg_list[i].meta[_keywords['shift_x']] = None
            reg_list[i].meta[_keywords['shift_y']] = None
            reg_list[i].meta[_keywords['rotation']] = None

    if clip_output:
        shifts = [(i.meta[_keywords['shift_x']], i.meta[_keywords['shift_y']])
                  if i.meta[_keywords['method']] != 'failed'
                  else (np.nan, np.nan)
                  for i in reg_list]
        shifts = np.array(shifts)
        xslice, yslice = _get_clip_slices(shifts, reg_list[0].shape)
        logger.info('Clipping output images to section: x=%s:%s, y=%s:%s',
                    xslice.start, xslice.stop, yslice.start, yslice.stop)
        for i in range(n):
            trim_image(reg_list[i], xslice, yslice, inplace=True)

    return reg_list
