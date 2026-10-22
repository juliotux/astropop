# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Background and noise estimation algorithms."""

from __future__ import annotations

import typing as _t
from .._typing import ImageLike as _ImageLike
from .._typing import NumberArray as _NumberArray

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


def background(
    data: _ImageLike,
    box_size: int | tuple[int, int]=64,
    filter_size: int | tuple[int, int]=3,
    mask: _ImageLike | None=None,
    bkg_method: _t.Literal['mode', 'mean', 'median']='mode',
    rms_method: _t.Literal['std', 'mad_std']='std',
    sigma_clip: float | tuple[float, float]=3.0,
    global_bkg: bool=True,
) -> tuple[_NumberArray, _NumberArray]:
    """Estimate the background and noise values in a image.

    The used algorithm is described by `photutils.background.Background2D`. It
    splits the image into boxes of size ``box_size`` and calculate the
    background and noise in each box. The final background and noise values
    are interpolated using ``filter_size`` neighboring boxes.

    Parameters
    ----------
    data
        2D array containing the image to extract the background.
    box_size
        Size of background boxes in pixels.
        Default: 64
    filter_size
        Filter size in boxes unit.
        Default: 3
    mask
        Boolean mask where 1 pixels are masked out in the background
        calculation.
        Default: `None`
    sigma_clip
        Sigma clipping value. If a tuple is given, the first value is the
        lower sigma and the second is the upper sigma.
    bkg_method
        Method to calculate the background. By ``'mode'`` it uses the
        `~photutils.background.SExtractorBackground` algorithm. ``'mean'`` and
        ``'median'`` uses `~photutils.background.MeanBackground` and
        `~photutils.background.MedianBackground` respectively.
        Default: ``'mode'``
    rms_method
        Method to calculate the rms. ``'std'`` uses the standard deviation
        and ``'mad_std'`` uses the median absolute deviation.
        Default: ``'std'``
    global_bkg
        If True, the algorithm returns a single value for background and rms,
        else, a 2D image with local values will be returned. The global value
        is computed as the median of the 2D local values.
        Default: True

    Returns
    -------
    background
        Background value or image.
    rms
        RMS value or image.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.photometry import background
    >>> pixels = np.full((16, 16), 100.)
    >>> sky, rms = background(pixels, box_size=8, filter_size=1)
    >>> print(float(sky), float(rms))
    100.0 0.0

    >>> # Request maps instead of the global median estimates.
    >>> sky_map, rms_map = background(pixels, box_size=8, global_bkg=False)
    >>> print(sky_map.shape, rms_map.shape)
    (16, 16) (16, 16)
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
                       bkg_rms_estimator=_rms_methods[rms_method]())

    if global_bkg:
        return float(bkg.background_median), float(bkg.background_rms_median)
    else:
        return np.array(bkg.background), np.array(bkg.background_rms)
