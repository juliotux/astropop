"""
This file implements a matplotlib Normalize object
which mimics the functionality of scaling functions in ds9

The transformation from data values to normalized (0-1) display
intensities are as follows:

- Data to normal:
   y = clip( (x - vmin) / (vmax - vmin), 0, 1)
- normal to warped: Apply a monotonic, non-linear scaling, that preserves
  the endpoints
- warped to greyscale:
  y = clip((x - bias) * contrast + 0.5, 0, 1)

This code is based on https://github.com/glue-viz/ds9norm
"""

# implementation details
# The relevant ds9 code is located at saotk/frame/colorscale.C and
# saotk/colorbar/colorbar.C
#
# As much as possible, we use verbose but inplace ufuncs to minimize
# temporary arrays

import numpy as np
from matplotlib.colors import Normalize


__all__ = ['DS9Normalize']


def fast_limits(data, plo, phi):
    """Quickly estimate percentiles in an array,
    using a downsampled version

    Parameters
    ----------
        data : array-like
            The data to be used
        plo : float
            Lo percentile (0-100)
        phi : float
            High percentile (0-100)

    Returns
    -------
        Tuple of floats.
            Approximate values of each percentile in data[component]
    """
    if ~np.isfinite(data).any():
        return (0.0, 1.0)

    data = data[np.isfinite(data)]
    lo, hi = np.percentile(data, [plo, phi])
    return lo, hi


def norm(x, vmin, vmax):
    """
    Linearly scale data between [vmin, vmax] to [0, 1]. Clip outliers
    """
    result = (x - 1.0 * vmin)
    result = np.divide(result, vmax - vmin, out=result)
    result = np.clip(result, 0, 1, out=result)
    return result


def cscale(x, bias, contrast):
    """
    Apply bias and contrast scaling. Overwrite input

    Parameters
    ----------
    x : array
      Values between 0 and 1
    bias : float
    contrast : float

    Returns
    -------
    The input x, scaled inplace
    """
    x = np.subtract(x, bias, out=x)
    x = np.multiply(x, contrast, out=x)
    x = np.add(x, 0.5, out=x)
    x = np.clip(x, 0, 1, out=x)
    return x


def linear_warp(x, vmin, vmax, bias, contrast):
    return cscale(norm(x, vmin, vmax), bias, contrast)


def log_warp(x, vmin, vmax, bias, contrast, exp=1000.0):
    x = norm(x, vmin, vmax)
    x = np.multiply(exp, x, out=x)
    # sidestep numpy bug that masks log(1)
    # when out is provided
    x = np.add(x, 1.001, out=x)
    x = np.log(x, out=x)
    x = np.divide(x, np.log(exp + 1.0), out=x)
    x = cscale(x, bias, contrast)
    return x


def pow_warp(x, vmin, vmax, bias, contrast, exp=1000.0):
    x = norm(x, vmin, vmax)
    x = np.power(exp, x, out=x)
    x = np.subtract(x, 1, out=x)
    x = np.divide(x, exp - 1)
    x = cscale(x, bias, contrast)
    return x


def sqrt_warp(x, vmin, vmax, bias, contrast):
    x = norm(x, vmin, vmax)
    x = np.sqrt(x, out=x)
    x = cscale(x, bias, contrast)
    return x


def squared_warp(x, vmin, vmax, bias, contrast):
    x = norm(x, vmin, vmax)
    x = np.power(x, 2, out=x)
    x = cscale(x, bias, contrast)
    return x


def asinh_warp(x, vmin, vmax, bias, contrast):
    x = norm(x, vmin, vmax)
    x = np.divide(np.arcsinh(np.multiply(x, 10, out=x), out=x), 3, out=x)
    x = cscale(x, bias, contrast)
    return x


warpers = dict(linear=linear_warp,
               log=log_warp,
               sqrt=sqrt_warp,
               power=pow_warp,
               squared=squared_warp,
               arcsinh=asinh_warp)


# for mpl <= 1.1, Normalize is an old-style class
# explicitly inheriting from object allows property to work
class DS9Normalize(Normalize, object):
    """
    A Matplotlib Normalize object that implements DS9's image stretching

    Parameters
    ----------
    stretch : 'linear' | 'log' | 'sqrt' | 'power' | 'squared' | 'arcsinh'
        Which stretch function to use. Defaults to 'linear'
    clip_lo : number
        Where to clip the minimum image intensity. Expressed as a percentile
        of the range of intensity values. Defaults to 0
    clip_hi : number
        Where to clip the maximum image intensity. Expressed as a percentile
        of the range of intensity values. Defaults to 100
    bias : float
        The location of the middle-grey value,
        relative to the [clip_lo, clip_hi] range. Defaults to 0.5
    contrast : float
        How quickly the scaled image transitions from black to white,
        relative to the [clip_lo, clip_hi] range. Defaults to 1.0
    """

    def __init__(self, stretch='linear',
                 bias=0.5, contrast=1.0,
                 clip_lo=0., clip_hi=100.):
        super(DS9Normalize, self).__init__()
        self.stretch = stretch
        self.bias = bias
        self.contrast = contrast
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi

    @property
    def clip_lo(self):
        return self._clip_lo

    @clip_lo.setter
    def clip_lo(self, value):
        self._clip_lo = value
        self.vmin = self.vmax = None

    @property
    def clip_hi(self):
        return self._clip_hi

    @clip_hi.setter
    def clip_hi(self, value):
        self._clip_hi = value
        self.vmin = self.vmax = None

    @property
    def stretch(self):
        return self._stretch

    @stretch.setter
    def stretch(self, value):
        if value not in warpers:
            raise ValueError("Invalid stretch: %s\n Valid options are: %s" %
                             (value, warpers.keys()))
        self._stretch = value

    def autoscale(self, A):
        self.update_clip(A)

    def autoscale_None(self, A):
        if self.vmin is None or self.vmax is None:
            self.update_clip(A)

    def update_clip(self, image):
        self.vmin, self.vmax = fast_limits(image, self.clip_lo, self.clip_hi)

    def __call__(self, value, clip=False):
        self.autoscale_None(value)

        inverted = self.vmax <= self.vmin

        hi, lo = max(self.vmin, self.vmax), min(self.vmin, self.vmax)

        warp = warpers[self.stretch]
        result = warp(value, lo, hi, self.bias, self.contrast)

        if inverted:
            result = np.subtract(1, result, out=result)

        result = np.ma.MaskedArray(result, copy=False)

        return result
