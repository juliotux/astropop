# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Compute shifts and translate astronomical images."""

import abc
from skimage.registration import phase_cross_correlation
from skimage import transform
import numpy as np

from ..logger import logger
from ..framedata import check_framedata, FrameData


__all__ = ['CrossCorrelationRegister', 'AsterismRegister',
           'register_framedata_list']


class _BaseRegister(abc.ABC):
    """Base class for Registers."""

    _name = None

    @staticmethod
    def _apply_transform_image(image, tform, cval=0):
        """Apply the transform to an image."""
        return transform.warp(image, tform, mode='constant', cval=cval,
                              preserve_range=True)

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
            reg_frame = FrameData(None)
            reg_frame.meta = frame2.meta

        reg_frame.data = data
        reg_frame.mask = mask

        if not frame2.uncertainty.empty:
            unct = frame2.get_uncertainty(return_none=False)
            unct = self._apply_transform_image(unct,
                                               tform, cval=np.nan)
            reg_frame.uncertainty = unct

        reg_frame.meta['astropop registration'] = self._name
        reg_frame.meta['astropop registration_shift'] = list(tform.translation)
        reg_frame.meta['astropop registration_rot'] = np.rad2deg(tform.rotation)

        if reg_frame.wcs is not None:
            logger.warn('WCS in frame2 is not None. Due to the transform it'
                        ' will be erased.')
            reg_frame.wcs = None

        return reg_frame


class CrossCorrelationRegister(_BaseRegister):
    """Register images usgin `~skimage.Register.phase_cross_correlation`.

    It uses cross-correlation to find a translation-only transform between
    two images. It obtains an initial estimate of the cross-correlation
    peak by an FFT and then refines the shift estimation by upsampling
    the DFT only in a small neighborhood of that estimate by means of a
    matrix-multiply DFT[1]_.

    References
    ----------
    .. [1] :DOI:`10.1364/OL.33.000156`
    """

    _name = 'cross-correlation'

    def __init__(self, upsample_factor=1, space='real'):
        """Initialize a CrossCorrelationRegister instance.

        Parameters
        ----------
        upsample_factor : int (optional)
            Upsampling image factor. Images will be Registered to within
            ``1 / upsample_factor`` of a pixel.
            Default: 1 (no upsampling)
        space : {'real', 'fourier'} (optional)
            Defines how the algorithm interprets input data. "real" means
            data will be FFT'd to compute the correlation, while "fourier"
            data will bypass FFT of input data. Case insensitive.
        """
        self._up_factor = upsample_factor
        self._fft_space = space

    def _compute_transform(self, image1, image2, mask1=None, mask2=None):
        # Masks are ignored by default
        dy, dx = phase_cross_correlation(image1, image2,
                                         upsample_factor=self._up_factor,
                                         space=self._fft_space,
                                         return_error=False)
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

    References
    ----------
    .. [1] :DOI:`10.1016/j.ascom.2020.100384`
    """

    _name = 'asterism-matching'

    def __init__(self, max_control_points=50, detection_threshold=5):
        """Initialize the AsterismRegister instance.

        Parameters
        ----------
        max_control_points : int, optional
            Maximum control points (stars) used in asterism matching.
            Default: 50
        detection_threshold : int, optional
            Minimum SNR detection threshold.
            Default: 5

        Raises
        ------
        ImportError: if astroalign is not installed
        """
        try:
            import astroalign
        except ImportError:
            raise ImportError('AsterismRegister requires astroalign tools.')

        from ..photometry import starfind, background

        self._aa = astroalign
        self._sf = starfind
        self._bkg = background
        self._max_cntl_pts = max_control_points
        self._threshold = detection_threshold

    def _compute_transform(self, image1, image2, mask1=None, mask2=None):
        if mask1 is not None or mask2 is not None:
            logger.info("Masks are ignored in ChiSqRegister.")

        # use our starfind to work with only good sources
        bkg, rms = self._bkg(image1, global_bkg=True)
        sources1 = self._sf(image1, self._threshold, bkg, rms)
        bkg, rms = self._bkg(image2, global_bkg=True)
        sources2 = self._sf(image2, self._threshold, bkg, rms)
        sources1.sort('flux', reverse=True)
        sources2.sort('flux', reverse=True)
        sources1 = np.array(list(zip(sources1['x'], sources1['y'])))
        sources2 = np.array(list(zip(sources2['x'], sources2['y'])))

        tform, ctl_pts = self._aa.find_transform(sources1, sources2,
                                                 self._max_cntl_pts)

        logger.debug("Asterism matching performed with sources at: "
                     "image1: %s; image2 %s",
                     ctl_pts[0], ctl_pts[1])

        return tform


def register_framedata_list(frame_list, algorithm='cross-correlation',
                            cval='median', inplace=False, **kwargs):
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
    cval : float or {'median', 'mean'} (optional)
        Fill value for the empty pixels in the transformed image. If 'mean' or
        'median', the correspondent values will be computed from the image.
        Default: 'median'
    inplace : bool (optional)
        Perform the operation inplace, modifying the original FrameData
        container. If `False`, a new container will be created.
        Default: `False`
    **kwargs :
        keyword arguments to be passed to `CrossCorrelationRegister` or
        `AsterismRegister` during instance creation. See the parameters in
        each class documentation.
    """
    # check the algorithms
    if algorithm == 'cross-correlation':
        reg = CrossCorrelationRegister(**kwargs)
    elif algorithm == 'asterism-matching':
        reg = AsterismRegister(**kwargs)
    else:
        raise ValueError(f'Algorithm {algorithm} unknown.')

    for i in frame_list:
        if not isinstance(i, FrameData):
            raise TypeError('Only a list of FrameData instances is allowed.')
        if i.shape != frame_list[0].shape:
            raise ValueError('Images with incompatible shapes. Only frames '
                             'with same shape allowed.')

    n = len(frame_list)
    reg_list = [None]*n
    for i in range(n):
        logger.info('Registering image %i from %i', i+1, n)
        reg_list[i] = reg.register_framedata(frame_list[0], frame_list[i],
                                             cval=cval, inplace=inplace)

    return reg_list
