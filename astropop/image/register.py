# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Compute shifts and translate astronomical images."""

import abc
from functools import partial
from skimage import transform
import numpy as np

from .processing import trim_image
from ..logger import logger
from ..framedata import check_framedata, FrameData


__all__ = ['CrossCorrelationRegister', 'AsterismRegister',
           'register_framedata_list', 'compute_shift_list']


def _algorithm_check(algorithm, kwargs):
    """Check the algorithms and get the register."""
    if algorithm == 'cross-correlation':
        return CrossCorrelationRegister(**kwargs)
    if algorithm == 'asterism-matching':
        return AsterismRegister(**kwargs)
    raise ValueError(f'Algorithm {algorithm} unknown.')


def _check_compatible_list(frame_list):
    """Check if the list if it is compatible with the register."""
    for i in frame_list:
        if not isinstance(i, FrameData):
            raise TypeError('Only a list of FrameData instances is allowed.')
        if i.shape != frame_list[0].shape:
            raise ValueError('Images with incompatible shapes '
                             f'{frame_list[0].shape} and {i.shape}.'
                             'Only frames with same shape allowed.')


def _get_clip_slices(shifts, imshape):
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

    @staticmethod
    def _apply_transform_image(image, tform, cval=0):
        """Apply the transform to an image."""
        return transform.warp(image, tform, mode='constant', cval=cval,
                              preserve_range=True)

    @abc.abstractmethod
    def _compute_transform(self, image1, image2, mask1=None, mask2=None):
        """Compute the transform to register image2 to image1."""

    def compute_transform(self, image1, image2, mask1=None, mask2=None):
        """Compute the transform to register image2 to image1.

        Parameters
        ----------
        image1 : 2d `~numpy.ndarray`
            Reference image for registration.
        image2 : 2d `~numpy.ndarray`
            Moving image for registration.
        mask1, mask2 : 2d `~numpy.ndarray` (optional)
            Masks for images 1 and 2.
            Default: `None`

        Returns
        -------
        tfrom : `~skimage.transform.AffineTransform`
            Transform computed to project image2 in image2.
        """
        return self._compute_transform(image1, image2, mask1, mask2)

    def register_image(self, image1, image2, mask1=None, mask2=None,
                       cval='median'):
        """Align and transform an image2 to match image1.

        Parameters
        ----------
        image1 : 2d `~numpy.ndarray`
            Reference image for registration.
        image2 : 2d `~numpy.ndarray`
            Moving image for registration.
        mask1, mask2 : 2d `~numpy.ndarray` (optional)
            Masks for images 1 and 2.
            Default: `None`
        cval : float or {'median', 'mean'} (optional)
            Fill value for transformed pixels from outside the image. If
            'median' or 'mean', these statistics will be computed as cval.
            Default: 'median'

        Returns
        -------
        reg_image : `~numpy.ndarray`
            Registered image according the transform computed by the class.
        mask : `~numpy.ndarray`
            Mask for the registered image.
        tfrom : `~skimage.transform.AffineTransform`
            Transform computed to project image2 in image2.
        """
        # equal images are just returned
        if np.all(image1 == image2):
            logger.info('Images are equal, skipping registering.')
            return image1, np.zeros_like(image1), transform.AffineTransform()

        tform = self._compute_transform(image1, image2, mask1, mask2)
        if mask2 is None:
            mask2 = np.zeros_like(image2)

        if cval == 'median':
            cval = np.nanmedian(image2)
        if cval == 'mean':
            cval = np.nanmean(image2)

        reg_image = self._apply_transform_image(image2, tform, cval=cval)
        logger.info('Filling registered image with cval=%.2f', cval)
        logger.info('Registering image with: '
                    'translation=%s, rotation=%.2f°',
                    tform.translation, np.rad2deg(tform.rotation))
        mask = self._apply_transform_image(mask2, tform, cval=1)
        mask = mask > 0
        return reg_image, mask, tform

    def register_framedata(self, frame1, frame2, cval='median',
                           inplace=False):
        """Align and transform a frame2 to match frame1.

        Parameters
        ----------
        frame1 : `~astropop.framedata.FrameData`
            Reference image for registration.
        image2 : `~astropop.framedata.FrameData`
            Moving image for registration.
        cval : `float` or {'median', 'mean'} (optional)
            Fill value for transformed pixels from outside the image. If
            'median' or 'mean', these statistics will be computed as cval.
            Default: 'median'
        inplace : `bool` (optional)
            Perform the operation in original frame2 instance. The original
            data will be changed.

        Returns
        -------
        reg_frame : `~astropop.framedata.FrameData`
            Registered frame.
        """
        frame1 = check_framedata(frame1)
        frame2 = check_framedata(frame2)

        im1 = np.array(frame1.data)
        im2 = np.array(frame2.data)
        msk1 = frame1.mask if frame1.mask is None else np.array(frame1.mask)
        msk2 = frame2.mask if frame2.mask is None else np.array(frame2.mask)

        data, mask, tform = self.register_image(im1, im2, msk1, msk2,
                                                cval=cval)

        if inplace:
            reg_frame = frame2
        else:
            # Copy the frame to mantain the memmap caching behavior
            reg_frame = frame2.copy()

        reg_frame.data = data
        reg_frame.mask = mask

        if not frame2.uncertainty.empty:
            unct = frame2.get_uncertainty(return_none=False)
            unct = self._apply_transform_image(unct,
                                               tform, cval=np.nan)
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
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images. Used only if one of ``reference_mask`` or
        ``moving_mask`` is not None.

    Notes
    -----
    - Due to a bug in the `~skimage.registration.phase_cross_correlation`
      normalization is automatically disabled.
    - ``return_error`` is set to ``'always'`` to avoid keep compatibility.

    References
    ----------
    .. [1] :doi:`10.1364/OL.33.000156`
    """

    _name = 'cross-correlation'

    def __init__(self, **kwargs):
        import skimage
        from skimage.registration import phase_cross_correlation
        if skimage.__version__ >= '0.19':
            kwargs['normalization'] = None
        kwargs['return_error'] = 'always'
        self._pcc = partial(phase_cross_correlation, **kwargs)

    def _compute_transform(self, image1, image2, mask1=None, mask2=None):
        if mask1 is not None or mask2 is not None:
            logger.debug("Masks are ignored in CrossCorrelationRegister.")
        # Masks are ignored by default
        dy, dx = self._pcc(image1, image2)[0]
        return transform.AffineTransform(translation=(-dx, -dy))


class AsterismRegister(_BaseRegister):
    """Register images using asterism matching. Based on astroalign [1]_.

    This Register algorith compute the transform between 2 images based
    on the position of detected sources. It can handle both translation and
    rotation of the images. It compare similar 3-points asterisms in 2
    images and find the best possible affine transform between them.

    This package requires `astroalign` to work. The main difference here to
    the bare astroalign Register is that we use our `starfind`
    implementation to find the sources, to keep just good punctual sources
    in the Register, and sort them by brightness, using the brighter sources
    in the work. This may allow a better result in the Register, according our
    experiments.

    Parameters
    ----------
    max_control_points : int, optional
        Maximum control points (stars) used in asterism matching.
        Default: 50
    detection_threshold : int, optional
        Minimum SNR detection threshold.
        Default: 5
    detection_function : {'sepfind', 'starfind', 'daofind'} (optional)
        Detection function to use.
        Default: 'sepfind'
    detection_kwargs : dict (optional)
        Keyword arguments to pass to the detection function.

    Raises
    ------
    ImportError: if astroalign is not installed

    References
    ----------
    .. [1] :doi:`10.1016/j.ascom.2020.100384`
    """

    _name = 'asterism-matching'

    def __init__(self, max_control_points=50, detection_threshold=5,
                 detection_function='sepfind', **detection_kwargs):
        try:
            import astroalign
        except ImportError:
            raise ImportError('AsterismRegister requires astroalign tools.')

        from ..photometry import sepfind, starfind, daofind, background

        funcs = {'sepfind': sepfind, 'starfind': starfind, 'daofind': daofind}
        self._aa = astroalign
        self._sf = partial(funcs[detection_function], **detection_kwargs)
        self._bkg = background
        self._max_cntl_pts = max_control_points
        self._threshold = detection_threshold

    def _compute_transform(self, image1, image2, mask1=None, mask2=None):
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


def compute_shift_list(frame_list, algorithm='cross-correlation',
                       ref_image=0, skip_failure=False, **kwargs):
    """Compute the shift between a list of frames.

    Parameters
    ----------
    frame_list : list
        A list containing `~astropop.framedata.FrameData` images to be
        registered. All images must have the same shape.
    algorithm : {'cross-correlation', 'asterism-matching'} (optional)
        The algorithm to compute the `~skimage.transform.AffineTransform`
        between the images.
        'cross-correlation' will compute the transform
        using `~skimage.transform.phase_cross_correlation` method.
        'asterism-matching' will use `~astroalign` to match asterisms of 3
        detected stars in the field and compute the transform.
        Default: 'cross-correlation'
    ref_image : int (optional)
        Reference image index to compute the registration.
        Default: 0
    skip_failure : bool (optional)
        If True, the images that fail to register will be skipped and their
        shifts will be set to nan.
        Default: False
    **kwargs :
        keyword arguments to be passed to `CrossCorrelationRegister` or
        `AsterismRegister` during instance creation. See the parameters in
        each class documentation.
    """
    reg = _algorithm_check(algorithm, kwargs)
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


def register_framedata_list(frame_list, algorithm='cross-correlation',
                            ref_image=0, clip_output=False,
                            cval='median', inplace=False, skip_failure=False,
                            **kwargs):
    """Perform registration in a framedata list.

    Parameters
    ----------
    frame_list : list
        A list containing `~astropop.framedata.FrameData` images to be
        registered. All images must have the same shape.
    algorith : {'cross-correlation', 'asterism-matching'} (optional)
        The algorithm to compute the `~skimage.transform.AffineTransform`
        between the images.
        'cross-correlation' will compute the transform
        using `~skimage.transform.phase_cross_correlation` method.
        'asterism-matching' will use `~astroalign` to match asterisms of 3
        detected stars in the field and compute the transform.
        Default: 'cross-correlation'
    ref_image : int (optional)
        Reference image index to compute the registration.
        Default: 0
    clip_output : bool (optional)
        If True, the output images will be clipped to a only-valid pixels
        frame.
    cval : float or {'median', 'mean'} (optional)
        Fill value for the empty pixels in the transformed image. If 'mean' or
        'median', the correspondent values will be computed from the image.
        Default: 'median'
    skip_failure: bool (optional)
        If True, the images that fail to register will be skipped. Their data
        will be fill with the cval and all pixels mask will be set to
        True. If False, the error will be raised.
        Default: False
    inplace : bool (optional)
        Perform the operation inplace, modifying the original FrameData
        container. If `False`, a new container will be created.
        Default: `False`
    **kwargs :
        keyword arguments to be passed to `CrossCorrelationRegister` or
        `AsterismRegister` during instance creation. See the parameters in
        each class documentation.
    """
    reg = _algorithm_check(algorithm, kwargs)
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
            reg_list[i].mask[:] = True
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
