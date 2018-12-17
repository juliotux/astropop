# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sep
import numpy as np

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.ndimage.filters import convolve

from ._utils import _sep_fix_byte_order
from ..math.models.moffat import moffat_r, moffat_fwhm, PSFMoffat2D
from ..math.models.gaussian import gaussian_r, gaussian_fwhm, PSFGaussian2D
from ..math.array import trim_array, xy2r
from ..logger import logger


def gen_filter_kernel(size):
    """Generate sextractor like filter kernels."""
    # TODO: better implement
    if size == 3:
        return np.array([[0,1,0],
                         [1,4,1],
                         [0,1,0]])
    elif size == 5:
        return np.array([[1,2,3,2,1],
                         [2,4,6,4,2],
                         [3,6,9,6,3],
                         [2,4,6,4,2],
                         [1,2,3,2,1]])
    elif size == 7:
        return np.array([[ 1, 2, 3, 4, 3, 2, 1],
                         [ 2, 4, 6, 8, 6, 4, 2],
                         [ 3, 6, 9,12, 9, 6, 3],
                         [ 4, 8,12,16,12, 8, 4],
                         [ 3, 6, 9,12, 9, 6, 3],
                         [ 2, 4, 6, 8, 6, 4, 2],
                         [ 1, 2, 3, 4, 3, 2, 1]])


def background(data, box_size, filter_size, mask=None, global_bkg=True):
    """Estimate the image background using SExtractor algorithm.

    If global_bkg, return a single value for background and rms, else, a 2D
    image with local values.
    """
    d = _sep_fix_byte_order(data)
    bkg = sep.Background(d, bw=box_size, bh=box_size,
                         fw=filter_size, fh=filter_size,
                         mask=mask)

    if global_bkg:
        return bkg.globalback, bkg.globalrms
    else:
        return bkg.back(), bkg.rms()


def sexfind(data, snr, background, noise, recenter=False,
            mask=None, fwhm=None, filter_kernel=3, **sep_kwargs):
    """Find sources using SExtractor segmentation algorithm.

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

    sources = sep.extract(d-background, snr, err=noise, mask=mask,
                          **sep_kwargs)

    if sep_kwargs.get('segmentation_map', False):
        sources, smap = sources
        return Table(sources), smap
    else:
        return Table(sources)


def daofind(data, snr, background, noise, fwhm, mask=None,
            sharp_limit=(0.2, 1.0),
            round_limit=(-1.0, 1.0)):
    """Find sources using DAOfind algorithm.

    Translated from IDL Astro package by D. Jones. Original function available
    at PythonPhot package. https://github.com/djones1040/PythonPhot
    The function recieved some improvements to work better
    """
    # Compute hmin based on snr, background and noise
    hmin = np.median(snr*noise)

    image = data.astype(np.float64) - background
    maxbox = 13 	#Maximum size of convolution box in pixels

    # Get information about the input image
    type = np.shape(image)
    if len(type) != 2:
        raise ValueError ('data array must be 2 dimensional')
    n_x  = type[1]
    n_y = type[0]
    logger.debug('Input Image Size is {}x{}'.format(n_x, n_y))

    if fwhm < 0.5:
        raise ValueError('Supplied FWHM must be at least 0.5 pixels')

    radius = np.max([0.637*fwhm, 2.001])
    radsq = radius**2
    nhalf = np.min([int(radius), int((maxbox-1)/2.)])
    nbox = 2*nhalf + 1	# number of pixels in side of convolution box
    middle = nhalf      # Index of central pixel
    # lastro = n_x - nhalf
    # lastcl = n_y - nhalf

    sigsq = (fwhm*gaussian_fwhm_to_sigma)**2
    mask = np.zeros([nbox,nbox], dtype='int8' )  # Mask identifies valid pixels in convolution box
    g = np.zeros([nbox,nbox])  # Gaussian convolution kernel
    # dd = np.arange(nbox-1,dtype='int') + 0.5 - middle # Constants need to compute ROUND
    # dd2 = dd**2

    row2 = (np.arange(nbox)-nhalf)**2
    for i in range(nhalf+1):
        temp = row2 + i**2
        g[nhalf-i,:] = temp
        g[nhalf+i,:] = temp

    g_row = np.where(g <= radsq)
    mask[g_row[0],g_row[1]] = 1 # MASK is complementary to SKIP in Stetson's Fortran
    good = np.where(mask)  # Value of c are now equal to distance to center
    pixels = len(good[0])

    #  Compute quantities for centroid computations that can be used for all stars
    g = np.exp(-0.5*g/sigsq)

    #  In fitting Gaussians to the marginal sums, pixels will arbitrarily be
    # assigned weights ranging from unity at the corners of the box to
    # NHALF^2 at the center (e.g. if NBOX = 5 or 7, the weights will be
    #
    #                                 1   2   3   4   3   2   1
    #      1   2   3   2   1          2   4   6   8   6   4   2
    #      2   4   6   4   2          3   6   9  12   9   6   3
    #      3   6   9   6   3          4   8  12  16  12   8   4
    #      2   4   6   4   2          3   6   9  12   9   6   3
    #      1   2   3   2   1          2   4   6   8   6   4   2
    #                                 1   2   3   4   3   2   1
    #
    # respectively).  This is done to desensitize the derived parameters to
    # possible neighboring, brighter stars.
    xwt = np.zeros([nbox,nbox])
    wt = nhalf - np.abs(np.arange(nbox)-nhalf ) + 1
    for i in range(nbox):
        xwt[i,:] = wt
    ywt = np.transpose(xwt)
    sgx = np.sum(g*xwt,1)
    p = np.sum(wt)
    sgy = np.sum(g*ywt,0)
    sumgx = np.sum(wt*sgy)
    sumgy = np.sum(wt*sgx)
    sumgsqy = np.sum(wt*sgy*sgy)
    sumgsqx = np.sum(wt*sgx*sgx)
    vec = nhalf - np.arange(nbox)
    dgdx = sgy*vec
    dgdy = sgx*vec
    sdgdxs = np.sum(wt*dgdx**2)
    sdgdx = np.sum(wt*dgdx)
    sdgdys = np.sum(wt*dgdy**2)
    sdgdy = np.sum(wt*dgdy)
    sgdgdx = np.sum(wt*sgy*dgdx)
    sgdgdy = np.sum(wt*sgx*dgdy)

    c = g*mask  # Convolution kernel now in c
    sumc = np.sum(c)
    sumcsq = np.sum(c**2) - sumc**2/pixels
    sumc = sumc/pixels
    c[good[0],good[1]] = (c[good[0],good[1]] - sumc)/sumcsq
    c1 = np.exp(-.5*row2/sigsq)
    sumc1 = np.sum(c1)/nbox
    sumc1sq = np.sum(c1**2) - sumc1
    c1 = (c1-sumc1)/sumc1sq

    logger.debug('RELATIVE ERROR computed from FWHM {}'
                 .format(np.sqrt(np.sum(c[good[0],good[1]]**2))))

    h = convolve(image, c)  # Convolve image with kernel "c"

    minh = np.min(h)
    h[:,0:nhalf] = minh
    h[:,n_x-nhalf:n_x] = minh
    h[0:nhalf,:] = minh
    h[n_y-nhalf:n_y-1,:] = minh

    mask[middle,middle] = 0	# From now on we exclude the central pixel
    pixels = pixels - 1     # so the number of valid pixels is reduced by 1
    good = np.where(mask)  # "good" identifies position of valid pixels
    xx = good[1] - middle  # x and y coordinate of valid pixels
    yy = good[0] - middle  # relative to the center
    index = np.where(h >= hmin)  #Valid image pixels are greater than hmin
    nfound = len(index)
    logger.debug('{} pixels above threshold'.format(nfound))

    if nfound == 0:  # Any maxima found?
        logger.warn('No maxima exceed input threshold of {}'.format(hmin))
        return

    for i in range(pixels):
        hy = index[0]+yy[i]; hx = index[1]+xx[i]
        hgood = np.where((hy < n_y) & (hx < n_x) & (hy >= 0) & (hx >= 0))[0]
        stars = np.where (np.greater_equal(h[index[0][hgood], index[1][hgood]],
                                           h[hy[hgood], hx[hgood]]))
        nfound = len(stars)
        if nfound == 0:  # Do valid local maxima exist?
            logger.warn('No maxima exceed input threshold of {}'.format(hmin))
            return
        index = np.array([index[0][hgood][stars],index[1][hgood][stars]])

    ix = index[1]  # X index of local maxima
    iy = index[0]  # Y index of local maxima
    ngood = len(index[0])
    logger.debug('{} local maxima located above threshold'.format(ngood))

    nstar = 0  # NSTAR counts all stars meeting selection criteria
    badround = 0
    badsharp = 0
    badcntrd = 0

    x = np.zeros(ngood)
    y = np.zeros(ngood)
    flux = np.zeros(ngood)
    sharp = np.zeros(ngood)
    roundness = np.zeros(ngood)

    #  Loop over star positions# compute statistics
    for i in range(ngood):
        temp = image[iy[i]-nhalf:iy[i]+nhalf+1,
                     ix[i]-nhalf:ix[i]+nhalf+1]
        d = h[iy[i],ix[i]]  # "d" is actual pixel intensity

        #  Compute Sharpness statistic
        sharp_limit = sorted(sharp_limit)
        sharp1 = (temp[middle,middle] - (np.sum(mask*temp))/pixels)/d
        if (sharp1 < sharp_limit[0]) or (sharp1 > sharp_limit[1]):
            badsharp = badsharp + 1
            continue #Does not meet sharpness criteria

        #   Compute Roundness statistic
        dx = np.sum( np.sum(temp,axis=0)*c1)
        dy = np.sum( np.sum(temp,axis=1)*c1)
        if (dx <= 0) or (dy <= 0):
            badround = badround + 1
            continue     # Cannot compute roundness
        round_limit = sorted(round_limit)
        around = 2*(dx-dy) / ( dx + dy )    #Roundness statistic
        if (around < round_limit[0]) or (around > round_limit[1]):
            badround = badround + 1
            continue     # Does not meet roundness criteria

        # Centroid computation: The centroid computation was modified in Mar 2008 and
        # now differs from DAOPHOT which multiplies the correction dx by 1/(1+abs(dx)).
        # The DAOPHOT method is more robust (e.g. two different sources will not merge)
        # especially in a package where the centroid will be subsequently be
        # redetermined using PSF fitting. However, it is less accurate, and introduces
        # biases in the centroid histogram. The change here is the same made in the
        # IRAF DAOFIND routine (see
        # http://iraf.net/article.php?story=7211;query=daofind )
        sd = np.sum(temp*ywt,axis=0)
        sumgd = np.sum(wt*sgy*sd)
        sumd = np.sum(wt*sd)
        sddgdx = np.sum(wt*sd*dgdx)
        hx = (sumgd - sumgx*sumd/p) / (sumgsqy - sumgx**2/p)

        # HX is the height of the best-fitting marginal Gaussian. If this is not
        # positive then the centroid does not make sense
        if (hx <= 0):
            badcntrd = badcntrd + 1
            continue
        skylvl = (sumd - hx*sumgx)/p
        dx = (sgdgdx - (sddgdx-sdgdx*(hx*sumgx + skylvl*p)))/(hx*sdgdxs/sigsq)
        if np.abs(dx) >= nhalf:
            badcntrd = badcntrd + 1
            continue
        xcen = ix[i] + dx    #X centroid in original array
        # Find Y centroid
        sd = np.sum(temp*xwt,axis=1)
        sumgd = np.sum(wt*sgx*sd)
        sumd = np.sum(wt*sd)
        sddgdy = np.sum(wt*sd*dgdy)
        hy = (sumgd - sumgy*sumd/p) / (sumgsqx - sumgy**2/p)
        if (hy <= 0):
            badcntrd = badcntrd + 1
            continue

        skylvl = (sumd - hy*sumgy)/p
        dy = (sgdgdy - (sddgdy-sdgdy*(hy*sumgy + skylvl*p)))/(hy*sdgdys/sigsq)
        if np.abs(dy) >= nhalf:
            badcntrd = badcntrd + 1
            continue
        ycen = iy[i] +dy    #Y centroid in original array
        #  This star has met all selection criteria.  Print out and save
        x[nstar] = xcen
        y[nstar] = ycen
        flux[nstar] = d
        sharp[nstar] = sharp1
        roundness[nstar] = around
        nstar = nstar+1

    nstar = nstar-1		#NSTAR is now the index of last star found

    logger.debug('{} sources rejected by SHARPNESS criteria'.format(badsharp))
    logger.debug('{} sources rejected by ROUNDNESS criteria'.format(badround))
    logger.debug('{} sources rejected by CENTROID  criteria'.format(badcntrd))

    if nstar < 0:
        return

    x = x[0:nstar+1]
    y = y[0:nstar+1]
    flux = flux[0:nstar+1]
    sharp = sharp[0:nstar+1]
    roundness = roundness[0:nstar+1]

    t = Table([x, y, flux, sharp, roundness],
              names=('x', 'y', 'flux', 'sharpness', 'roundness'))

    return t


def starfind(data, snr, background, noise, fwhm, mask=None, box_size=35,
            sharp_limit=(0.2, 1.0), round_limit=(-1.0, 1.0)):
    """Find stars using daofind AND sexfind."""
    # First, we identify the sources with sexfind (fwhm independent)
    sources = sexfind(data, snr, background, noise, mask=mask,
                      fwhm=fwhm)
    # We compute the median FWHM and perform a optimum daofind extraction
    fwhm = calc_fwhm(data, sources['x'], sources['y'], box_size=box_size,
                     model='gaussian', min_fwhm=fwhm) or fwhm

    sources = daofind(data, snr, background, noise, fwhm, mask=mask,
                      sharp_limit=sharp_limit, round_limit=round_limit)
    sources.meta['astropop fwhm'] = fwhm
    return sources


def sources_mask(shape, x, y, a, b, theta, mask=None, scale=1.0):
    """Create a mask to cover all sources."""
    image = np.zeros(shape, dtype=bool)
    sep.mask_ellipse(image, x, y, a, b, theta, r=scale)
    if mask is not None:
        image |= np.array(mask)
    return image


def _fwhm_loop(model, data, x, y, xc, yc):
    # FIXME: with curve fitting, gaussian model goes crazy
    if model == 'gaussian':
        model = gaussian_r
        mfwhm = gaussian_fwhm
        p0 = (1.0, np.max(data), np.min(data))
    elif model == 'moffat':
        model = moffat_r
        mfwhm = moffat_fwhm
        p0 = (1.0, 1.5, np.max(data), np.min(data))
    else:
        raise ValueError('Model {} not available.'.format(model))
    r, f = xy2r(x, y, data, xc, yc)
    args = np.argsort(r)
    try:
        popt, pcov = curve_fit(model, r[args], f[args], p0=p0)
        return mfwhm(*popt[:-2])
    except:
        return np.nan


def calc_fwhm(data, x, y, box_size=25, model='gaussian', min_fwhm=3.0):
    """Calculate the median FWHM of the image with Gaussian or Moffat fit."""
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
    if model == 'gaussian':
        model = PSFGaussian2D(x_0=xc, y_0=yc)
    elif model == 'moffat':
        model = PSFMoffat2D(x_0=xc, y_0=yc)
    else:
        raise ValueError('Model {} not available.'.format(model))
    m_fit = fitter(model, x, y, data)
    return m_fit.x_0.value, m_fit.y_0.value


def recenter_sources(data, x, y, box_size=25, model='gaussian'):
    """Recenter teh sources using a PSF model."""
    indices = np.indices(data.shape)
    rects = [trim_array(data, box_size, (xi, yi), indices=indices)
             for xi, yi in zip(x, y)]
    fitter = LevMarLSQFitter()
    coords = [_recenter_loop(fitter, model, d[0], d[1], d[2], xi, yi)
              for d, xi, yi in zip(rects, x, y)]
    coords = np.array(coords)
    nx = coords[:,0]
    ny = coords[:,1]
    return nx, ny
