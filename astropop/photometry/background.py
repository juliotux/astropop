# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Background and noise estimation algorithms."""

import numpy as np
from photutils.background import Background2D, SExtractorBackground, \
                                 MeanBackground, MedianBackground, \
                                 StdBackgroundRMS, MADStdBackgroundRMS
from astropy.stats import SigmaClip


__all__ = ['background']


_bkg_methods = {
    'mode': SExtractorBackground,
    'mean': MeanBackground,
    'median': MedianBackground
}

_rms_methods = {
    'std': StdBackgroundRMS,
    'mad_std': MADStdBackgroundRMS
}


def background(data, box_size=64, filter_size=3, mask=None,
               bkg_method='mode', rms_method='std', sigma_clip=3.0,
               global_bkg=True):
    """Estimate the background and noise values in a image.

    The used algorithm is described by `photutils.background.Background2D`. It
    splits the image into boxes of size `box_size` and calculate the
    background and noise in each box. The final background and noise values
    are interpolated using `filter_size` neighboring boxes.

    Parameters
    ----------
    data: array_like
        2D array containing the image to extract the background.
    box_size: `int` (optional)
        Size of background boxes in pixels.
        Default: 64
    filter_size: `int` (optional)
        Filter size in boxes unit.
        Default: 3
    mask: array_like (optional)
        Boolean mask where 1 pixels are masked out in the background
        calculation.
        Default: `None`
    sigma_clip: `float` or `tuple` (optional)
        Sigma clipping value. If a tuple is given, the first value is the
        lower sigma and the second is the upper sigma.
    bkg_method: ``'mode'``, ``'mean'``, ``'median'`` (optional)
        Method to calculate the background. By ``'mode'`` it uses the
        `~photutils.background.SExtractorBackground` algorithm. ``'mean'`` and
        ``'median'`` uses `~photutils.background.MeanBackground` and
        `~photutils.background.MedianBackground` respectively.
        Default: ``'mode'``
    rms_method: ``'std'``, ``'mad_std'`` (optional)
        Method to calculate the rms. ``'std'`` uses the standard deviation
        and ``'mad_std'`` uses the median absolute deviation.
        Default: ``'std'``
    global_bkg: `bool`
        If True, the algorithm returns a single value for background and rms,
        else, a 2D image with local values will be returned. The global value
        is computed as the median of the 2D local values.
        Default: True

    Returns
    -------
    background: `float` or `~numpy.ndarray`
        Background value or image.
    rms: `float` or `~numpy.ndarray`
        RMS value or image.
    """
    data = np.array(data)
    # check valid methods
    if bkg_method not in _bkg_methods:
        raise ValueError(f"Unknown background method: {bkg_method}")
    if rms_method not in _rms_methods:
        raise ValueError(f"Unknown rms method: {rms_method}")

    # create the sigma clipper
    if isinstance(sigma_clip, (tuple, list)):
        sclip = SigmaClip(sigma_lower=sigma_clip[0],
                          sigma_upper=sigma_clip[1])
    else:
        sclip = SigmaClip(sigma=sigma_clip)

    # create the background object
    bkg = Background2D(data, box_size=box_size, filter_size=filter_size,
                       mask=mask, sigma_clip=sclip,
                       bkg_estimator=_bkg_methods[bkg_method](),
                       bkgrms_estimator=_rms_methods[rms_method]())

    if global_bkg:
        return float(bkg.background_median), float(bkg.background_rms_median)
    else:
        return np.array(bkg.background), np.array(bkg.background_rms)
