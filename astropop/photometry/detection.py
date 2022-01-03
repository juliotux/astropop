# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sep
import numpy as np
import warnings

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.ndimage.filters import convolve

from ._utils import _sep_fix_byte_order
from ..math.moffat import moffat_r, moffat_fwhm, PSFMoffat2D
from ..math.gaussian import gaussian_r, gaussian_fwhm, PSFGaussian2D
from ..math.array import trim_array, xy2r
from ..logger import logger


def gen_filter_kernel(size):
    """Generate sextractor like filter kernels."""
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
    bkg = sep.Background(d, bw=box_size, bh=box_size,
                         fw=filter_size, fh=filter_size,
                         mask=mask)

    if global_bkg:
        return bkg.globalback, bkg.globalrms
    return bkg.back(), bkg.rms()


def sepfind(data, threshold, background, noise,
            mask=None, fwhm=None, filter_kernel=3,
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
    fwhm: `int` (optional)
        Full width at half maximum, fwhm = 2.35*sigma
        Default: `None`
    filter_kernel: `int` (optional)
        Combined filter, which can provide optimal signal-to-noise
        detection for objects with some known shape
        Default: `3`

    sep_kwargs can be any kwargs to be passed to sep.extract function.
    """
    d = _sep_fix_byte_order(data)

    # Check if need to create a new kernel
    if np.isscalar(filter_kernel):
        filter_kernel = gen_filter_kernel(filter_kernel)

    sep_kwargs['filter_kernel'] = filter_kernel

    # Compute min area based on FWHM
    if 'minarea' not in sep_kwargs.keys() and fwhm is not None:
        sep_kwargs['minarea'] = int((3.14*fwhm**2)/4)

    sources = sep.extract(d-background, threshold, err=noise, mask=mask,
                          **sep_kwargs)

    if sep_kwargs.get('segmentation_map', False):
        sources, smap = sources
        return Table(sources), smap

    return Table(sources)


class DAOFind:
    """Use DAOFind method to detect punctual sources.

    Parameters
    ----------
    fwhm: `int` or `float`
        Default gaussian FWHM for convolution kernel
    sharp_limit: array_like or `None` (optional)
        Low and high cutoff for the sharpness statistic.
        `None` will disable the sharpness filtering.
        Default: (0.2, 1.0)
    round_limit : array_like or `None` (optional)
        Low and high cutoff for the roundness statistic.
        `None` will disable the roundness filtering.
        Default: (-1.0,1.0)

    Notes
    -----
    - Adapted from IDL Astro package by D. Jones. Original function available
      at PythonPhot package. https://github.com/djones1040/PythonPhot
      The function recieved some improvements to work better.
    - For our roundness statistics, we use themaximum value between the
      symmetry based roundness (`SROUND` in IRAF DAOFIND) and the marginal
      gaussian fit roundness (`GROUND` in IRAF DAOFIND), allowing better
      identification of assymetric sources in diagonal.
    """
    _maxbox = 13  # Maximum convolution box

    def __init__(self, fwhm, sharp_limit=(0.2, 1.0), round_limit=(-1.0, 1.0)):
        if fwhm < 0.5:
            raise ValueError('Supplied FWHM must be at least 0.5 pixels')
        # fhwm and related stuff
        self._fwhm = fwhm
        self._sharp_limit = sharp_limit
        self._round_limit = round_limit

        self._compute_convolution_kernel()
        self._compute_constants()

    def _compute_convolution_kernel(self):
        """Generate the proper gaussian kernel."""
        self._sigma2 = (self._fwhm*gaussian_fwhm_to_sigma)**2
        # radius=1.5*sigma, greater then 2
        self._radius = np.max([0.637*self._fwhm, 2.001])
        # index of the center of convolution box
        self._nhalf = np.min([int(self._radius),
                              int((self._maxbox-1)/2.)])
        # size of convolution box. Automatic less or equal maxbox
        self._nbox = 2*self._nhalf + 1
        shape = (self._nbox, self._nbox)

        # valid pixels in convolution kernel, Stetson 'mask'
        self._conv_mask = np.zeros(shape, dtype='int8')

        y, x = np.indices(shape)
        rsq = (x-self._nhalf)**2 + (y-self._nhalf)**2
        self._good = np.where(rsq <= self._radius**2)
        pixels = len(self._good[0])
        self._conv_mask[self._good] = 1
        # convolution kernel, Stetson 'g'
        self._g = np.exp(-0.5*rsq/self._sigma2)
        self._kernel = self._g*self._conv_mask

        # Normalize the convolution kernel and zero not good pixels
        sumc = np.sum(self._kernel)
        sumcsq = np.sum(self._kernel**2) - sumc**2/pixels
        sumc = sumc/pixels
        self._kernel[self._good] = (self._kernel[self._good] - sumc)/sumcsq

        logger.debug('Relative error computed from FWHM: %f',
                     np.sqrt(np.sum(self._kernel[self._good]**2)))

    def _compute_constants(self):
        """Compute some constants needed for statistics."""
        _, x = np.indices((self._nbox, self._nbox))

        self._wt = self._nhalf - np.abs(np.arange(self._nbox)-self._nhalf) + 1
        self._vec = self._nhalf - np.arange(self._nbox)
        self._p = np.sum(self._wt)

        self._c1 = (np.arange(self._nbox)-self._nhalf)**2
        self._c1 = np.exp(-0.5*self._c1/self._sigma2)
        sumc1 = np.sum(self._c1)/self._nbox
        self._c1 = (self._c1-sumc1)/(np.sum(self._c1**2) - sumc1)

        self._xwt = self._nhalf - np.abs(x-self._nhalf) + 1
        self._ywt = np.transpose(self._xwt)

        self._sgx = np.sum(self._g*self._xwt, 1)
        self._sgy = np.sum(self._g*self._ywt, 0)

        self._dgdx = self._sgy*self._vec
        self._dgdy = self._sgx*self._vec

        self._sumgx = np.sum(self._wt*self._sgy)
        self._sumgy = np.sum(self._wt*self._sgx)

        self._sumgsqx = np.sum(self._wt*self._sgx*self._sgx)
        self._sumgsqy = np.sum(self._wt*self._sgy*self._sgy)

        self._sdgdx = np.sum(self._wt*self._dgdx)
        self._sdgdy = np.sum(self._wt*self._dgdy)

        self._sdgdxs = np.sum(self._wt*self._dgdx**2)
        self._sdgdys = np.sum(self._wt*self._dgdy**2)

        self._sgdgdx = np.sum(self._wt*self._sgy*self._dgdx)
        self._sgdgdy = np.sum(self._wt*self._sgx*self._dgdy)

    def _convolve_image(self, image, n_x, n_y):
        """Convolve image with gaussian kernel."""
        h = convolve(image, self._kernel)

        minh = np.min(h)
        nhalf = self._nhalf

        # Fix borders
        h[:, 0:nhalf] = minh
        h[:, n_x-nhalf:n_x] = minh
        h[0:nhalf, :] = minh
        h[n_y-nhalf:n_y - 1, :] = minh

        return h

    def _find_peaks(self, image, index, n_x, n_y):
        """Find peaks above threshold."""
        mask = self._conv_mask.copy()
        mask[self._nhalf, self._nhalf] = 0  # exclude central pixel
        pixels = np.sum(mask)
        good = np.where(mask)
        xx = good[1] - self._nhalf
        yy = good[0] - self._nhalf

        for i in range(pixels):
            hy = index[0]+yy[i]
            hx = index[1]+xx[i]
            # only inside image
            hgood = np.where((hy < n_y) & (hy >= 0) &
                             (hx < n_x) & (hx >= 0))[0]
            stars = np.where(np.greater_equal(image[index[0][hgood],
                                                    index[1][hgood]],
                                              image[hy[hgood],
                                                    hx[hgood]]))
            if len(stars) == 0:
                logger.error('No maxima exceed input threshold.')
                return
            index = np.array([index[0][hgood][stars],
                              index[1][hgood][stars]])

        logger.debug('%i localmaxima located above threshold.',
                     len(index[0]))
        return index

    def _calc_ground(self, chunk):
        """Compute the gaussian marginal fit roundness."""
        dx = np.sum(np.sum(chunk, axis=0)*self._c1)
        dy = np.sum(np.sum(chunk, axis=1)*self._c1)
        if dx <= 0 or dy <= 0:  # invalid roundness
            return np.nan
        return 2*(dx-dy)/(dx+dy)

    def _calc_sround(self, chunk):
        """Compute the folding roundness."""
        # Quads
        # 3 3 4 4 4
        # 3 3 4 4 4
        # 3 3 x 1 1
        # 2 2 2 1 1
        # 2 2 2 1 1
        nhalf = self._nhalf
        chunk = np.array(chunk)
        chunk[nhalf, nhalf] = 0  # copy and setcentral pixel to 0

        quad1 = chunk[nhalf:, nhalf+1:]
        quad2 = chunk[nhalf+1:, :nhalf+1]
        quad3 = chunk[:nhalf+1, :nhalf]
        quad4 = chunk[:nhalf, nhalf:]

        sum2 = -quad1.sum()+quad2.sum()-quad3.sum()+quad4.sum()
        sum4 = chunk.sum()

        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            sround = 2.0 * sum2 / sum4
        return sround

    def _calc_sharp(self, chunk, d, mask):
        """Compute the sharpness."""
        center = chunk[self._nhalf, self._nhalf]
        sides = (np.sum(mask*chunk)/np.sum(mask))
        return (center-sides)/d

    def _calc_centroid(self, chunk, ix, iy):
        """Compute the centroid"""
        # Notes from D.Jones:
        # Modified version in Mar 2008 that differ from original DAOPHOT,
        # which multiplies the correction dx by 1/(1+abs(dx)).
        # DAOPHOT method is more robust (e.g. two different sources will
        # not merge).
        # This method is more accurate and do not introduce biases in the
        # centroid histogram.
        # The change was also applyed in IRAF DAFIND routine.
        # (see http://iraf.net/article.php?story=7211;query=daofind)
        # x centroid
        sd = np.sum(chunk*self._ywt, axis=0)
        sumgd = np.sum(self._wt*self._sgy*sd)
        sumd = np.sum(self._wt*sd)
        sddgdx = np.sum(self._wt*sd*self._dgdx)
        # hx is the height of the best-fitting marginal gaussian
        hx = (sumgd-self._sumgx*sumd/self._p)
        hx /= (self._sumgsqy-self._sumgx**2/self._p)
        if hx <= 0:
            return
        skylvl = (sumd - hx*self._sumgx)/self._p
        dx = sddgdx-self._sdgdx*(hx*self._sumgx + skylvl*self._p)
        dx = (self._sgdgdx-dx)/(hx*self._sdgdxs/self._sigma2)
        if np.abs(dx) >= self._nhalf:
            return

        # y centroid
        sd = np.sum(chunk*self._xwt, axis=1)
        sumgd = np.sum(self._wt*self._sgx*sd)
        sumd = np.sum(self._wt*sd)
        sddgdy = np.sum(self._wt*sd*self._dgdy)
        hy = (sumgd - self._sumgy*sumd/self._p)
        hy /= (self._sumgsqx - self._sumgy**2/self._p)
        if hy <= 0:
            return
        skylvl = (sumd - hy*self._sumgy)/self._p
        dy = sddgdy-self._sdgdy*(hy*self._sumgy + skylvl*self._p)
        dy = (self._sgdgdy-dy)/(hy*self._sdgdys/self._sigma2)
        if np.abs(dy) >= self._nhalf:
            return

        return ix+dx, iy+dy

    def _compute_statistics(self, image, image_convolved, index):
        """Compute source statistics for each star."""
        h = image_convolved
        ngood = len(index[0])
        iy, ix = index
        nhalf = self._nhalf
        mask = self._conv_mask.copy()
        mask[nhalf, nhalf] = 0  # exclude central pixel

        x = np.full(ngood, fill_value=np.nan, dtype='f8')
        y = np.full(ngood, fill_value=np.nan, dtype='f8')
        flux = np.full(ngood, fill_value=np.nan, dtype='f8')
        sharpness = np.full(ngood, fill_value=np.nan, dtype='f8')
        roundness = np.full(ngood, fill_value=np.nan, dtype='f8')

        # loop over all stars
        for i in range(ngood):
            # original image chunk
            temp = image[iy[i]-nhalf:iy[i]+nhalf+1,
                         ix[i]-nhalf:ix[i]+nhalf+1]
            # convolved image chunk
            temp_conv = h[iy[i]-nhalf:iy[i]+nhalf+1,
                          ix[i]-nhalf:ix[i]+nhalf+1]
            # convolved central pixel intensity
            d = h[iy[i], ix[i]]

            # compute sharpness
            sharpness[i] = self._calc_sharp(temp, d, mask)

            # compute roundness
            # maximum value between symmetry and fit roundness
            roundness[i] = np.max([self._calc_ground(temp),
                                   self._calc_sround(temp_conv)])
            if roundness[i] == np.nan:
                continue  # invalid source

            # compute centroid
            centroids = self._calc_centroid(temp, ix[i], iy[i])
            if centroids is not None:
                x[i], y[i] = centroids

        return Table({'x': x, 'y': y, 'flux': flux,
                      'sharp': sharpness, 'round': roundness})

    def _filter_sources(self, sources):
        """Perform roundness, sharpness and centroid filtering."""
        # filter stars by sharpness
        sharp_rej = 0
        sharpmask = False
        if self._sharp_limit is not None:
            sharp_limit = sorted(self._sharp_limit)
            sharpmask = np.isnan(sources['sharp'])
            sharpmask |= sources['sharp'] < sharp_limit[0]
            sharpmask |= sources['sharp'] > sharp_limit[1]
            sharp_rej = np.sum(sharpmask)
            logger.debug('%i sources rejected by sharpness',
                         sharp_rej)

        # filter by roundness
        round_rej = 0
        roundmask = False
        if self._round_limit is not None:
            round_limit = sorted(self._round_limit)
            roundmask = np.isnan(sources['round'])
            roundmask |= sources['round'] < round_limit[0]
            roundmask |= sources['round'] > round_limit[1]
            round_rej = np.sum(roundmask) - sharp_rej
            logger.debug('%i sources rejected by roundness',
                         round_rej)

        centroidmask = np.isnan(sources['x'])
        centroidmask |= np.isnan(sources['y'])
        logger.debug('%i sources rejected by invalid centroid',
                     np.sum(centroidmask) - round_rej - sharp_rej)

        mask = sharpmask & roundmask & centroidmask
        return sources[~mask]

    def find_stars(self, data, threshold, background, noise):
        """Find stars in a single image using daofind."""
        if len(np.shape(data)) != 2:
            raise ValueError('Data array must be 2 dimensional.')

        n_y, n_x = np.shape(data)
        hmin = np.median(threshold*noise)
        image = np.array(data, dtype=np.float64) - background

        h = self._convolve_image(image, n_x, n_y)

        index = np.where(h >= hmin)
        if len(index[0]) == 0:
            logger.error('No maxima exceed input threshold of %f', hmin)
            return
        logger.debug('Found %i pixels above threshold', len(index[0]))

        index = self._find_peaks(h, index, n_x, n_y)
        t = self._compute_statistics(image, h, index)
        return self._filter_sources(t)


def daofind(data, threshold, background, noise, fwhm, mask=None,
            sharp_limit=(0.2, 1.0), round_limit=(-1.0, 1.0)):
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
    mask: array_like (optional)
        Boolean mask where 1 pixels are masked out in the background
        calculation.
        Default: `None`
    sharp_limit: array_like or `None` (optional)
        Low and high cutoff for the sharpness statistic.
        `None` will disable the sharpness filtering.
        Default: (0.2, 1.0)
    round_limit : array_like or `None` (optional)
        Low and high cutoff for the roundness statistic.
        `None` will disable the roundness filtering.
        Default: (-1.0,1.0)
    """
    # TODO: mask is ignored in the original algorith
    # Find a way to use it
    d = DAOFind(fwhm, sharp_limit, round_limit)
    return d.find_stars(data, threshold, background, noise)


def starfind(data, threshold, background, noise, fwhm, mask=None, box_size=35,
             sharp_limit=(0.2, 1.0), round_limit=(-1.0, 1.0)):
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
    fwhm: `float`
        Full width at half maximum: to be used in the convolve filter.
    mask: `bool` (optional)
        Boolean mask where 1 pixels are masked out in the background
        calculation.
        Default: `None`
    box_size: `int` (optional)
        Size of background boxes in pixels.
        Default: 35
    sharp_limit: array_like (optional)
        Low and high cutoff for the sharpness statistic.
        Default: (0.2, 1.0)
    round_limit : array_like (optional)
        Low and high cutoff for the roundness statistic.
        Default: (-1.0,1.0)
    """
    # First, we identify the sources with sepfind (fwhm independent)
    sources = sepfind(data, threshold, background, noise, mask=mask,
                      fwhm=fwhm)
    # We compute the median FWHM and perform a optimum daofind extraction
    fwhm = calc_fwhm(data, sources['x'], sources['y'], box_size=box_size,
                     model='gaussian', min_fwhm=fwhm) or fwhm

    sources = daofind(data, threshold, background, noise, fwhm, mask=mask,
                      sharp_limit=sharp_limit, round_limit=round_limit)
    sources.meta['astropop fwhm'] = fwhm
    return sources


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
        Size of background boxes in pixels.
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


def _recenter_loop(fitter, model, data, x, y, xc, yc):
    """
    Parameters
    ----------
    fitter: `~astropy.modeling.fitting.Fitter`
        Fitter based on the Levenberg-Marquardt algorithm and
        least squares statistic.
    model: {`gaussian`, `moffat`}
        Choose a Gaussian or Moffat model.
    data: array_like
        2D array containing the image to extract the source.
    x, y: array_like
        x and y indexes ofthe pixels in the image.
    xc, yc: array_like
        x and y initial guess positions of the source.
    """
    if model == 'gaussian':
        model = PSFGaussian2D(x_0=xc, y_0=yc)
    elif model == 'moffat':
        model = PSFMoffat2D(x_0=xc, y_0=yc)
    else:
        raise ValueError(f'Model {model} not available.')
    m_fit = fitter(model, x, y, data)
    return m_fit.x_0.value, m_fit.y_0.value


def recenter_sources(data, x, y, box_size=25, model='gaussian'):
    """Recenter teh sources using a PSF model.

    Parameters
    ----------
    data: array_like
        2D array containing the image to extract the source.
    x, y: array_like
        x, y centroid position.
    box_size: `int` (optional)
        Size of background boxes in pixels.
        Default: 25
    model: `str`
        Choose a Gaussian or Moffat model.
        Default: `gausiann`
    """
    indices = np.indices(data.shape)
    rects = [trim_array(data, box_size, (xi, yi), indices=indices)
             for xi, yi in zip(x, y)]
    fitter = LevMarLSQFitter()
    coords = [_recenter_loop(fitter, model, d[0], d[1], d[2], xi, yi)
              for d, xi, yi in zip(rects, x, y)]
    coords = np.array(coords)
    nx = coords[:, 0]
    ny = coords[:, 1]
    return nx, ny
