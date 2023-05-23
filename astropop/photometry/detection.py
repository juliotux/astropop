# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.convolution import convolve
from astropy.nddata.utils import overlap_slices
from astropy.table import Table
from scipy.optimize import curve_fit
from photutils.detection import DAOStarFinder
from photutils.morphology import data_properties
from photutils.segmentation import SourceFinder, SourceCatalog, \
                                   make_2dgaussian_kernel

from astropop.math.moffat import moffat_r, moffat_fwhm
from astropop.math.gaussian import gaussian_r, gaussian_fwhm
from astropop.math.array import trim_array, xy2r


__all__ = ['segfind', 'daofind', 'starfind', 'median_fwhm']


_default_morfology_columns = ['segment_flux', 'fwhm', 'eccentricity',
                              'ellipticity', 'elongation', 'cxx', 'cyy', 'cxy',
                              'semimajor_sigma', 'semiminor_sigma',
                              'orientation']


@np.deprecate(message='This function will be removed before v1.0',
              new_name='segmentation_find')
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

    sep_kwargs can be any kwargs to be passed to sep.extract function.
    """
    from ._utils import _sep_fix_byte_order
    import sep

    def gen_filter_kernel(size):
        if size == 3:
            return np.array([[0, 1, 0],
                             [1, 4, 1],
                             [0, 1, 0]])
        if size == 5:
            return np.array([[1, 2, 3, 2, 1],
                             [2, 4, 6, 4, 2],
                             [3, 6, 9, 6, 3],
                             [2, 4, 6, 4, 2],
                             [1, 2, 3, 2, 1]])
        if size == 7:
            return np.array([[1, 2, 3, 4, 3, 2, 1],
                             [2, 4, 6, 8, 6, 4, 2],
                             [3, 6, 9, 12, 9, 6, 3],
                             [4, 8, 12, 16, 12, 8, 4],
                             [3, 6, 9, 12, 9, 6, 3],
                             [2, 4, 6, 8, 6, 4, 2],
                             [1, 2, 3, 4, 3, 2, 1]])
        return

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


def segfind(data, threshold, background, noise, mask=None, fwhm=None, npix=5,
            deblend=True):
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
    finder = SourceFinder(npixels=npix, deblend=deblend, progress_bar=False)
    sources = SourceCatalog(data, finder(conv_data, threshold, mask=mask),
                            convolved_data=conv_data, error=noise)

    # reorganize the table using the keywords of daofind
    res = Table()
    res['id'] = sources.label
    res['x'] = sources.xcentroid
    res['y'] = sources.ycentroid
    res['peak'] = sources.max_value
    res['flux'] = sources.segment_flux
    res['fwhm'] = sources.fwhm
    res['sigma_x'] = sources.semimajor_sigma
    res['sigma_y'] = sources.semiminor_sigma
    res['theta'] = sources.orientation
    res['eccentricity'] = sources.eccentricity
    res['elongation'] = sources.elongation
    res['ellipticity'] = sources.ellipticity
    res['cxx'] = sources.cxx
    res['cyy'] = sources.cyy
    res['cxy'] = sources.cxy
    res['area'] = sources.area
    res['label'] = sources.label

    # sort brightest first
    res.sort('flux', reverse=True)
    res['id'] = np.arange(len(res))+1

    return res


def daofind(data, threshold, background, noise, fwhm,
            mask=None, sharp_limit=(0.2, 1.0), round_limit=(-1.0, 1.0),
            exclude_border=True, positions=None):
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
    # Get the parameters for sharpness and roundness
    if sharp_limit is None:
        sharplo, sharphi = None, None
    else:
        sharplo, sharphi = sharp_limit
    if round_limit is None:
        roundlo, roundhi = None, None
    else:
        roundlo, roundhi = round_limit

    # DaoStarFinder uses absolute threshold value
    thresh = np.median(threshold * noise)

    dao = DAOStarFinder(thresh, fwhm=fwhm, sky=background,
                        sharplo=sharplo, sharphi=sharphi,
                        roundlo=roundlo, roundhi=roundhi,
                        exclude_border=exclude_border,
                        xycoords=positions)
    sources = dao(data, mask=mask)
    # additional filtering steps?

    morfology = sources_morfology(data, sources['xcentroid'],
                                  sources['ycentroid'],
                                  r=fwhm,
                                  background=background, mask=mask,
                                  columns=_default_morfology_columns)

    # reorganize the table using more standard keywords
    res = Table()
    res['id'] = sources['id']
    res['x'] = sources['xcentroid']
    res['y'] = sources['ycentroid']
    res['xcentroid'] = sources['xcentroid']
    res['ycentroid'] = sources['ycentroid']
    res['peak'] = sources['peak']
    res['flux'] = morfology['segment_flux']
    res['fwhm'] = morfology['fwhm']
    res['sharpness'] = sources['sharpness']
    r_arr = np.absolute([sources['roundness1'],
                         sources['roundness2']]).transpose()
    r_arg = np.argmax(r_arr, axis=1)
    res['roundness'] = r_arr[np.arange(len(r_arg)), r_arg]
    res['s_roundness'] = sources['roundness1']
    res['g_roundness'] = sources['roundness2']
    res['eccentricity'] = morfology['eccentricity']
    res['elongation'] = morfology['elongation']
    res['ellipticity'] = morfology['ellipticity']
    res['cxx'] = morfology['cxx']
    res['cyy'] = morfology['cyy']
    res['cxy'] = morfology['cxy']

    # reorder the sources by flux
    res.sort('flux', reverse=True)  # Sort the results by flux.

    res.sort('flux', reverse=True)
    res['id'] = np.arange(len(res))+1

    return res


def starfind(data, threshold, background, noise, fwhm=None, mask=None,
             sharp_limit=(0.2, 1.0), round_limit=(-1.0, 1.0),
             exclude_border=True):
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

    # Perform daofind using the optimal median fwhm
    s = daofind(data, threshold, background, noise, 1.5*fwhm, mask=mask,
                sharp_limit=sharp_limit, round_limit=round_limit,
                exclude_border=exclude_border)
    s.meta['astropop fwhm'] = fwhm
    return s


def sources_morfology(data, x, y, r, background=None, mask=None, columns=None):
    """Compute sources morfogoly.

    This function creates a fake `~photutils.segmentation.SegmentationImage`
    considering only circular sources centered on the (x,y) positions and
    generates a `~photutils.segmentation.SourceCatalog` using it. This class is
    useful to compute various parameters of the source morfology.

    Parameters
    ----------
    data: array_like
        2D array containing the image to extract the sources.
    x, y: array_like
        x and y coordinates of the source centroid
    r: `int`
        Aperture radius. The box size will be 2*r+1.
    background: `float` or `~numpy.ndarray` (optional)
        Background level estimation. Can be a single global value, or a 2D
        array with by-pixel values.
        Default: `None`
    mask: array_like (optional)
        Boolean mask where 1 pixels are masked out in the background
        calculation.
        Default: `None`
    columns: list (optional)
        List of columns to be returned. If `None`, only the default columns
        will be returned. See `~photutils.segmentation.SourceCatalog` for
        available columns.
        Default: `None`

    Returns
    -------
    properties: `~photutils.morfology.SourceCatalog`
        Source morfology properties.
    """
    shape = data.shape
    box = (2*r+1, 2*r+1)
    slices = [overlap_slices(shape, box, position=(yi, xi), mode='partial')[0]
              for xi, yi in zip(x, y)]

    # background must be a 2D array
    if np.isscalar(background) and background is not None:
        background = np.ones_like(data) * background

    for i in background, mask:
        if i is not None:
            if i.shape != shape:
                raise ValueError('data, background and mask must have the '
                                 'same shape.')

    columns = columns or _default_morfology_columns

    props = None
    for sly, slx in slices:
        d = data[sly, slx]
        if background is not None:
            bkg = background[sly, slx]
            d -= bkg
        else:
            bkg = None
        if mask is not None:
            msk = mask[sly, slx]

        else:
            msk = None
        t = data_properties(d, mask=msk, background=bkg)
        if props is None:
            props = t.to_table(columns)
        else:
            props.add_row(t.to_table(columns)[0])

    return props


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


def median_fwhm(data, x, y, box_size=25, model='gaussian', min_fwhm=3.0):
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


@np.deprecate(message='This function will be removed. Use median_fwhm.')
def calc_fwhm(*args, **kwargs):
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
    return median_fwhm(*args, **kwargs)
