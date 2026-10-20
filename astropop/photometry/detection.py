# Licensed under a 3-clause BSD style license - see LICENSE.rst


from __future__ import annotations

import typing as _t
import numpy.typing as _npt
from .._typing import NumberArray as _NumberArray
import numpy as np
from astropy.convolution import convolve
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata.utils import overlap_slices
from astropy.table import Table, vstack
from photutils.detection import DAOStarFinder
from photutils.segmentation import SourceFinder, SourceCatalog, SegmentationImage, \
                                   make_2dgaussian_kernel

from astropop.math.models import PSFMoffatRadial, PSFGaussianRadial
from astropop.math.array import trim_array, xy2r


__all__ = ['segfind', 'daofind', 'starfind', 'median_fwhm']


def segfind(
    data: _npt.ArrayLike,
    threshold: float,
    background: _NumberArray,
    noise: _NumberArray,
    mask: _npt.ArrayLike | None=None,
    fwhm: float | None=None,
    npix: int=5,
    deblend: bool=True,
) -> Table:
    """Find sources using `~photutils.segmentation`.

    Parameters
    ----------
    data: array_like
        2D array containing the image to extract the sources.
    threshold: `int` or `float`
        Minumim number of standard deviations above the background level where
        the algorithm will start to consider the sources.
    background: `float` or `~numpy.ndarray`
        Background level estimation. Can be a single global value, or a 2D
        array with by-pixel values.
    noise: `float` or `~numpy.ndarray`
        RMS or any other noise estimation to use in the detection. Can be a
        single global value, or a 2D array with by-pixel values.
    mask: array_like (optional)
        Boolean mask where 1 pixels are masked out in the background
        calculation.
        Default: `None`
    fwhm: `float` (optional)
        FWHM to generate the convolution kernel. If `None`, no convolution
        will be performed.
        Default: `None`
    npix: `int` (optional)
        Minimum number of connected pixels to consider a source.
        Default: ``5``
    deblend: `bool` (optional)
        If `True`, the algorithm will try to deblend the sources.
        Default: `True`

    Returns
    -------
    sources: `astropy.table.Table`
        Table with the sources found. The table will contain the following
        columns:

        - ``id``: source ID
        - ``x``, ``y``: x and y coordinates of the source centroid
        - ``xcentroid``, ``ycentroid``: same as ``x`` and ``y``
        - ``peak``: value of the source peak
        - ``flux``: integrated (background-subtracted) source flux within the
          segment
        - ``sigma_x``, ``sigma_y``: standard deviation of the source elipsoid
          along x and y
        - ``theta``: rotation angle of the source elipsoid (degrees) from the
          positive x axis
        - ``fwhm``: FWHM of circular gaussian fit of the source
        - ``eccentricity``: The eccentricity of the 2D Gaussian function that
          has the same second-order moments as the source.
        - ``elongation``: The ratio of the lengths of the semimajor and
          semiminor axes.
        - ``ellipticity``: 1.0 minus the elongation.
        - ``cxx``, ``cyy``, ``cxy``: SourceExtractor ellipse parameters.
          See [SourceExtractor docs]_
        - ``area``: area of the segment, in pixels
        - ``label``: label identifying the segment to which the source belongs.
          Can be used as y-sorted index.

    References
    ----------
    .. [SourceExtractor docs] Ellipse parameters
       (https://sextractor.readthedocs.io/en/latest/Position.html\
        #ellipse-parameters-cxx-cyy-cxy)
    """
    # algorithm needs the sky-subtracted image
    # also, force create a new instance of image
    data = data-background
    conv_data = data

    # perform the detection on convolved data
    if fwhm is not None:
        ksize = int(np.max([np.ceil(2*fwhm)+1, 3]))
        kernel = make_2dgaussian_kernel(fwhm=fwhm, size=ksize)
        conv_data = convolve(data, kernel, mask=mask, normalize_kernel=True)

    # algorithm needs the absolute value threshold
    threshold = threshold*noise
    if np.isscalar(noise):
        noise = np.ones_like(data)*noise

    # run the segmentation algorithm
    finder = SourceFinder(n_pixels=npix, deblend=deblend)
    seg_img = finder(conv_data, threshold, mask=mask)
    if seg_img is None:
        raise ValueError('No sources found')
    sources = SourceCatalog(data, seg_img,
                            convolved_data=conv_data, error=noise)

    # reorganize the table using the keywords of daofind
    res = Table()
    res['id'] = sources.label
    res['x'] = sources.x_centroid
    res['y'] = sources.y_centroid
    res['peak'] = sources.max_value
    res['flux'] = sources.segment_flux
    res['fwhm'] = sources.fwhm
    res['sigma_x'] = sources.semimajor_axis
    res['sigma_y'] = sources.semiminor_axis
    res['theta'] = sources.orientation
    res['eccentricity'] = sources.eccentricity
    res['elongation'] = sources.elongation
    res['ellipticity'] = sources.ellipticity
    res['cxx'] = sources.ellipse_cxx
    res['cyy'] = sources.ellipse_cyy
    res['cxy'] = sources.ellipse_cxy
    res['area'] = sources.area
    res['label'] = sources.label

    # sort brightest first
    res.sort('flux', reverse=True)
    res['id'] = np.arange(len(res))+1

    return res


def _dao_source_morphology(
    data: _npt.ArrayLike,
    x: _npt.ArrayLike,
    y: _npt.ArrayLike,
    r: float,
    mask: _npt.ArrayLike | None=None,
) -> Table:
    """Measure each DAO source within its own circular segmentation mask."""
    columns = ['segment_flux', 'fwhm', 'eccentricity', 'elongation',
               'ellipticity', 'ellipse_cxx', 'ellipse_cyy', 'ellipse_cxy']
    tables = []
    for xi, yi in zip(x, y):
        slices = overlap_slices(data.shape, (2*r+1, 2*r+1),
                                position=(yi, xi), mode='partial')[0]
        cutout = data[slices]
        yy, xx = np.indices(cutout.shape)
        distance_sq = (xx + slices[1].start - xi)**2
        distance_sq += (yy + slices[0].start - yi)**2
        # Separate catalogs preserve overlapping apertures: a shared label
        # image would assign their common pixels to only one of the sources.
        segment = SegmentationImage((distance_sq < r**2).astype(int))
        cutout_mask = None if mask is None else mask[slices]
        catalog = SourceCatalog(cutout, segment, mask=cutout_mask)
        tables.append(catalog.to_table(columns=columns))
    return vstack(tables, metadata_conflicts='silent')


def daofind(
    data: _npt.ArrayLike,
    threshold: float,
    background: _NumberArray,
    noise: _NumberArray,
    fwhm: float,
    mask: _npt.ArrayLike | None=None,
    sharp_limit: tuple[float, float] | None=(0.2, 1.0),
    round_limit: tuple[float, float] | None=(-1.0, 1.0),
    exclude_border: bool=True,
    positions: _npt.ArrayLike | None=None,
) -> Table:
    """Find sources using DAOfind algorithm.

    Parameters
    ----------
    data: array_like
        2D array containing the image to extract the sources.
    threshold: `int` or `float`
        Minumim number of standard deviations above the background level where
        the algorithm will start to consider the sources.
    background: `float` or `~numpy.ndarray`
        Background level estimation. Can be a single global value, or a 2D
        array with by-pixel values.
    noise: `float` or `~numpy.ndarray`
        RMS or any other noise estimation to use in the detection. Can be a
        single global value, or a 2D array with by-pixel values. Due to
        limitations of photutils, if a 2D array is provided, it will be
        converted to a scalar using the median.
    fwhm: `float`
        Full width at half maximum: to be used in the convolve filter.
    sharp_limit: array_like or `None` (optional)
        Low and high cutoff for the sharpness statistic. `None` will disable
        the sharpness filtering.
        Default: (0.2, 1.0)
    round_limit : array_like or `None` (optional)
        Low and high cutoff for the roundness statistic. `None` will disable
        the roundness filtering.
        Default: (-1.0,1.0)
    mask: array_like (optional)
        Boolean mask where 1 pixels are masked out in the background
        calculation.
        Default: `None`
    exclude_border: `bool` (optional)
        If `True`, sources found within half the size of the convolution
        kernel from the image borders are excluded.
        Default: `True`
    positions: array_like (optional)
        List of (x, y) positions where to look for sources. If provided, the
        source finding step will be ignored and the centroids will be refined
        and filtered.
        Default: `None`

    Returns
    -------
    sources: `astropy.table.Table`
        Table with the sources found. The table will contain the following
        columns:

        - ``id``: source ID
        - ``x``, ``y``: x and y coordinates of the source centroid
        - ``xcentroid``, ``ycentroid``: same as ``x`` and ``y``
        - ``peak``: value of the source peak
        - ``flux``: integrated (background-subtracted) source flux within the
          segment
        - ``fwhm``: FWHM of circular gaussian fit of the source
        - ``sharpness``: The source DAOFIND sharpness statistic.
        - ``roundness``: The source DAOFIND roundness statistic. Maximum
          between ``g_roundness`` and ``r_roundness``.
        - ``g_roundness``: The DAOFIND roundness statistic based on Gaussian
          marginal fit. Good for assymetries aligned with the x or y axis.
        - ``s_roundness``: The DAOFIND roundness statistic based on symmetry
          of the source. Good for assymetries aligned with the diagonal.
        - ``sigma_x``, ``sigma_y``: standard deviation of the source elipsoid
          along x and y
        - ``theta``: rotation angle of the source elipsoid (degrees) from the
          positive x axis
        - ``fwhm``: FWHM of circular gaussian fit of the source
        - ``eccentricity``: The eccentricity of the 2D Gaussian function that
          has the same second-order moments as the source.
        - ``elongation``: The ratio of the lengths of the semimajor and
          semiminor axes.
        - ``ellipticity``: 1.0 minus the elongation.
        - ``cxx``, ``cyy``, ``cxy``: SourceExtractor ellipse parameters.
          See [SourceExtractor docs]_
        - ``area``: area of the segment, in pixels
    """
    # always use background subtracted data
    d = data-background

    # DaoStarFinder uses absolute threshold value
    thresh = np.median(threshold * noise)

    dao = DAOStarFinder(thresh, fwhm=fwhm,
                        sharpness_range=sharp_limit,
                        roundness_range=round_limit,
                        exclude_border=exclude_border,
                        xycoords=positions)
    sources = dao(d, mask=mask)
    # additional filtering steps?

    catalog = _dao_source_morphology(d,
                                   sources['x_centroid'],
                                   sources['y_centroid'],
                                   fwhm,
                                   mask=mask)

    # reorganize the table using more standard keywords
    res = Table()
    res['id'] = sources['id']
    res['x'] = sources['x_centroid']
    res['y'] = sources['y_centroid']
    res['xcentroid'] = sources['x_centroid']
    res['ycentroid'] = sources['y_centroid']
    res['peak'] = sources['peak']
    res['flux'] = catalog['segment_flux']
    res['fwhm'] = catalog['fwhm']
    res['sharpness'] = sources['sharpness']
    r_arr = np.absolute([sources['roundness1'],
                         sources['roundness2']]).transpose()
    r_arg = np.argmax(r_arr, axis=1)
    res['roundness'] = r_arr[np.arange(len(r_arg)), r_arg]
    res['s_roundness'] = sources['roundness1']
    res['g_roundness'] = sources['roundness2']
    res['eccentricity'] = catalog['eccentricity']
    res['elongation'] = catalog['elongation']
    res['ellipticity'] = catalog['ellipticity']
    res['cxx'] = catalog['ellipse_cxx']
    res['cyy'] = catalog['ellipse_cyy']
    res['cxy'] = catalog['ellipse_cxy']

    # reorder the sources by flux
    res.sort('flux', reverse=True)  # Sort the results by flux.
    res['id'] = np.arange(len(res))+1

    return res


def starfind(
    data: _npt.ArrayLike,
    threshold: float,
    background: _NumberArray,
    noise: _NumberArray,
    fwhm: float | None=None,
    mask: _npt.ArrayLike | None=None,
    sharp_limit: tuple[float, float] | None=(0.2, 1.0),
    round_limit: tuple[float, float] | None=(-1.0, 1.0),
    exclude_border: bool=True,
) -> Table:
    """Find stars using daofind AND sepfind.

    Parameters
    ----------
    data: array_like
        2D array containing the image to extract the sources.
    threshold: `int` or `float`
        Minumim number of standard deviations above the background level where
        the algorithm will start to consider the sources.
    background: `float` or `~numpy.ndarray`
        Background level estimation. Can be a single global value, or a 2D
        array with by-pixel values.
    noise: `float` or `~numpy.ndarray`
        RMS or any other noise estimation to use in the detection. Can be a
        single global value, or a 2D array with by-pixel values. Due to
        limitations of photutils, if a 2D array is provided, it will be
        converted to a scalar using the median.
    fwhm: `float` (optional)
        Initial guess of the FWHM to be used in the convolve filter. No need to
        be precise. Will be recomputed in the function.
        Default: `None`
    sharp_limit: array_like or `None` (optional)
        Low and high cutoff for the sharpness statistic. `None` will disable
        the sharpness filtering.
        Default: (0.2, 1.0)
    round_limit : array_like or `None` (optional)
        Low and high cutoff for the roundness statistic. `None` will disable
        the roundness filtering.
        Default: (-1.0,1.0)
    mask: array_like (optional)
        Boolean mask where 1 pixels are masked out in the background
        calculation.
        Default: `None`
    exclude_border: `bool` (optional)
        If `True`, sources found within half the size of the convolution
        kernel from the image borders are excluded.
        Default: `True`

    Returns
    -------
    sources: `astropy.table.Table`
        Table with the sources found. The table will contain the following
        columns:

        - ``id``: source ID
        - ``x``, ``y``: x and y coordinates of the source centroid
        - ``xcentroid``, ``ycentroid``: same as ``x`` and ``y``
        - ``peak``: value of the source peak
        - ``flux``: integrated (background-subtracted) source flux within the
          segment
        - ``fwhm``: FWHM of circular gaussian fit of the source
        - ``sharpness``: The source DAOFIND sharpness statistic.
        - ``roundness``: The source DAOFIND roundness statistic. Maximum
          between ``g_roundness`` and ``r_roundness``.
        - ``g_roundness``: The DAOFIND roundness statistic based on Gaussian
          marginal fit. Good for assymetries aligned with the x or y axis.
        - ``s_roundness``: The DAOFIND roundness statistic based on symmetry
          of the source. Good for assymetries aligned with the diagonal.
        - ``sigma_x``, ``sigma_y``: standard deviation of the source elipsoid
          along x and y
        - ``theta``: rotation angle of the source elipsoid (degrees) from the
          positive x axis
        - ``fwhm``: FWHM of circular gaussian fit of the source
        - ``eccentricity``: The eccentricity of the 2D Gaussian function that
          has the same second-order moments as the source.
        - ``elongation``: The ratio of the lengths of the semimajor and
          semiminor axes.
        - ``ellipticity``: 1.0 minus the elongation.
        - ``cxx``, ``cyy``, ``cxy``: SourceExtractor ellipse parameters.
          See [SourceExtractor docs]_
        - ``area``: area of the segment, in pixels
        - ``label``: label identifying the segment to which the source belongs.
          Can be used as y-sorted index.
    """
    # First, we identify the sources with sepfind (fwhm independent)
    sources = segfind(data, threshold, background, noise, mask=mask)
    fwhm = np.median(sources['fwhm'])  # get the median fwhm from the sources
    # we need to recompute, as the values getting returned seems to be
    # underestimated
    fwhm = median_fwhm(data, sources['x'], sources['y'], 5*fwhm,
                       model='gaussian')

    # Perform daofind using the optimal median fwhm
    s = daofind(data, threshold, background, noise, 1.5*fwhm, mask=mask,
                sharp_limit=sharp_limit, round_limit=round_limit,
                exclude_border=exclude_border)
    s.meta['astropop fwhm'] = fwhm
    return s


def _fwhm_loop(
    model: _t.Literal['gaussian', 'moffat'],
    data: _npt.ArrayLike,
    x: _npt.ArrayLike,
    y: _npt.ArrayLike,
    xc: float,
    yc: float,
) -> float:
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

    Returns
    -------
    fwhm: `float`
        FWHM of the source.

    Notes
    -----
    The sigma of the gaussian is capped to 100 pixels. The same as the Moffat
    width.
    """
    if model == 'gaussian':
        m = PSFGaussianRadial(sigma=1, flux=np.max(data), sky=np.min(data),
                              bounds={'sigma': (0.01, 100),
                                      'sky': (np.min(data), np.max(data))})
    elif model == 'moffat':
        m = PSFMoffatRadial(width=1, power=1.5, flux=np.max(data),
                            sky=np.min(data),
                            bounds={'width': (0.01, 100),
                                    'power': (1.01, 10),
                                    'sky': (np.min(data), np.max(data))})
    else:
        raise ValueError(f'Model {model} not available.')
    fitter = LevMarLSQFitter()
    r, f = xy2r(x, y, data, xc, yc)
    args = np.argsort(r)
    try:
        m_fit = fitter(m, r[args], f[args])
        return m_fit.fwhm
    except Exception:
        return np.nan


def median_fwhm(
    data: _npt.ArrayLike,
    x: _npt.ArrayLike,
    y: _npt.ArrayLike,
    box_size: int=25,
    model: _t.Literal['gaussian', 'moffat']='gaussian',
    min_fwhm: float=3.0,
) -> float:
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
        Default: ``gaussian``
    min_fwhm: float
        Minimum value for FWHM.
        Default: 3.0

    Returns
    -------
    fwhm : float
        Median fitted source FWHM in pixels. Nonfinite fits are ignored by
        the median. If the median is nonfinite or below ``min_fwhm``, returns
        ``min_fwhm``. Empty or all-invalid fits may emit a NumPy warning.

    Raises
    ------
    ValueError
        If ``model`` is neither ``'gaussian'`` nor ``'moffat'`` and a fit is
        attempted.
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


def calc_fwhm(*args: _t.Any, **kwargs: _t.Any) -> float:
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
        Default: ``gaussian``
    min_fwhm: float
        Minimum value for FWHM.
        Default: 3.0
    """
    import warnings
    warnings.warn('This function will be removed. Use median_fwhm.',
                  DeprecationWarning)
    return median_fwhm(*args, **kwargs)
