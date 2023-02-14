# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sep
import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit
from photutils.detection import DAOStarFinder

from ._utils import _sep_fix_byte_order
from ..math.moffat import moffat_r, moffat_fwhm
from ..math.gaussian import gaussian_r, gaussian_fwhm
from ..math.array import trim_array, xy2r


_default_sharp = (0.2, 1.0)
_default_round = (-1.0, 1.0)


def background(data, box_size=64, filter_size=3, mask=None,
               global_bkg=True):
    """Estimate the image background using SExtractor algorithm.

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
    global_bkg: `bool`
        If True, the algorithm returns a single value for background
        and rms, else, a 2D image with local values will be returned.
    """
    d = _sep_fix_byte_order(data)
    if mask is not None:
        mask = np.array(mask, dtype=bool)

    bkg = sep.Background(d, bw=box_size, bh=box_size,
                         fw=filter_size, fh=filter_size,
                         mask=mask)

    if global_bkg:
        return bkg.globalback, bkg.globalrms
    return bkg.back(), bkg.rms()


def sepfind(data, threshold, background, noise,
            mask=None, filter_kernel=3,
            **sep_kwargs):
    """Find sources using SExtractor segmentation algorithm.

    Parameters
    ----------
    data: array_like
        2D array containing the image to extract the source.
    threshold: `float`
        Minimum signal noise desired.
    background: `float`
        Background estimation.
    noise: `float`
        Root-mean-square at each point.
    mask: array_like (optional)
        Boolean mask where 1 pixels are masked out in the background
        calculation.
        Default: `None`
    filter_kernel: `int` (optional)
        Combined filter, which can provide optimal signal-to-noise
        detection for objects with some known shape
        Default: `3`
    **sep_kwargs: (optional)
        sep_kwargs can be any kwargs to be passed to sep.extract function.
        See https://sep.readthedocs.io/en/v1.0.x/api.html#sep.extract

    Returns
    -------
    sources: `astropy.table.Table`
        Table with the sources found.
    """
    d = _sep_fix_byte_order(data)
    if mask is not None:
        mask = np.array(mask, dtype=bool)

    # Check if need to create a new kernel
    if np.isscalar(filter_kernel):
        filter_kernel = gen_filter_kernel(filter_kernel)

    sep_kwargs['filter_kernel'] = filter_kernel

    sources = sep.extract(d-background, threshold, err=noise, mask=mask,
                          **sep_kwargs)

    if sep_kwargs.get('segmentation_map', False):
        sources = sources[0]  # ignore smap

    return Table(sources)


def daofind(data, threshold, background, noise, fwhm,
            mask=None, **kwargs):
    """Find sources using DAOfind algorithm.

    Parameters
    ----------
    data: array_like
        2D array containing the image to extract the source.
    threshold: `int`
        Minimum signal noise desired.
    background: `int`
        Background estimation.
    noise: `int`
        Root-mean-square at each point.
    fwhm: `int`
        Full width at half maximum: to be used in the convolve filter.
    sharp_limit: array_like or `None` (optional)
        Low and high cutoff for the sharpness statistic.
        `None` will disable the sharpness filtering.
        Default: (0.2, 1.0)
    round_limit : array_like or `None` (optional)
        Low and high cutoff for the roundness statistic.
        `None` will disable the roundness filtering.
        Default: (-1.0,1.0)

    Returns
    -------
    sources: `astropy.table.Table`
        Table with the sources found.

    """


def starfind(data, threshold, background, noise, fwhm=None,
             **kwargs):
    """Find stars using daofind AND sepfind.

    Parameters
    ----------
    data: array_like
        2D array containing the image to extract the source.
    threshold: `int`
        Minimum signal noise desired.
    background: `int`
        Background estimation.
    noise: `int`
        Root-mean-square at each point.
    fwhm: `float` (optional)
        Full width at half maximum to be used in the convolve filter.
        No need to be precise and will be recomputed in the function.
        Default: `None`
    mask: `bool` (optional)
        Boolean mask where 1 pixels are masked out in the background
        calculation.
        Default: `None`
    sharp_limit: array_like (optional)
        Low and high cutoff for the sharpness statistic.
        Default: (0.2, 1.0)
    round_limit : array_like (optional)
        Low and high cutoff for the roundness statistic.
        Default: (-1.0,1.0)
    skip_invalid_centroid: `bool`
        If `True`, the code will skip the invalid centroid verification and
        include sources with wrong centroid in the final list.
        Default: `False`
    """
    # First, we identify the sources with sepfind (fwhm independent)
    mask = kwargs.get('mask')
    sources = sepfind(data, threshold, background, noise, mask=mask)

    # We compute the median FWHM and perform a optimum daofind extraction
    box_size = 3*fwhm if fwhm is not None else 15  # 3xFWHM seems to be enough
    min_fwhm = fwhm or 2.0  # hardcoded 3.0 seems to be ok for stars
    fwhm = calc_fwhm(data, sources['x'], sources['y'], box_size=box_size,
                     model='gaussian', min_fwhm=min_fwhm)

    s = daofind(data, threshold, background, noise, fwhm, mask=mask,
                sharp_limit=kwargs.get('sharp_limit', _default_sharp),
                round_limit=kwargs.get('round_limit', _default_round),
                skip_invalid_centroid=kwargs.get('skip_invalid_centroid', 0))
    s.meta['astropop fwhm'] = fwhm
    return s


def sources_mask(shape, x, y, a, b, theta, mask=None, scale=1.0):
    """Create a mask to cover all sources.

    Parameters
    ----------
    shape: int or tuple of ints
        Shape of the new array.
    x,y: array_like
        Center of ellipse(s).
    a, b, theta: array_like (optional)
        Parameters defining the extent of the ellipe(s).
    mask: numpy.ndarray (optional)
        An optional mask.
        Default: `None`
    scale: array_like (optional)
        Scale factor of ellipse(s).
        Default: 1.0
    """
    image = np.zeros(shape, dtype=bool)
    sep.mask_ellipse(image, x, y, a, b, theta, r=scale)
    if mask is not None:
        image |= np.array(mask)
    return image


def _fwhm_loop(model, data, x, y, xc, yc):
    # FIXME: with curve fitting, gaussian model goes crazy
    """
    Parameters
    ----------
    model: `str`
        Choose a Gaussian or Moffat model.
    data: array_like
        2D array containing the image to extract the source.
    x, y: array_like
        x and y indexes ofthe pixels in the image.
    xc, yc: array_like
        x and y initial guess positions of the source.
    """
    if model == 'gaussian':
        model = gaussian_r
        mfwhm = gaussian_fwhm
        p0 = (1.0, np.max(data), np.min(data))
    elif model == 'moffat':
        model = moffat_r
        mfwhm = moffat_fwhm
        p0 = (1.0, 1.5, np.max(data), np.min(data))
    else:
        raise ValueError(f'Model {model} not available.')
    r, f = xy2r(x, y, data, xc, yc)
    args = np.argsort(r)
    try:
        popt, _ = curve_fit(model, r[args], f[args], p0=p0)
        return mfwhm(*popt[:-2])
    except Exception:
        return np.nan


def calc_fwhm(data, x, y, box_size=25, model='gaussian', min_fwhm=3.0):
    """Calculate the median FWHM of the image with Gaussian or Moffat fit.

    Parameters
    ----------
    data: array_like
        2D array containing the image to extract the source.
    x, y: array_like
        x, y centroid position.
    box_size: `int` (optional)
        Size of the box, in pixels, to fit the model.
        Default: 25
    model: {`gaussian`, `moffat`}
        Choose a Gaussian or Moffat model.
        Default: `gausiann`
    min_fwhm: float
        Minimum value for FWHM.
        Default: 3.0
    """
    indices = np.indices(data.shape)
    rects = [trim_array(data, box_size, (xi, yi), indices=indices)
             for xi, yi in zip(x, y)]
    fwhm = [_fwhm_loop(model, d[0], d[1], d[2], xi, yi)
            for d, xi, yi in zip(rects, x, y)]
    fwhm = np.nanmedian(fwhm)
    if fwhm < min_fwhm or ~np.isfinite(fwhm):
        fwhm = min_fwhm

    return fwhm
