# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Compute polarimetry of dual beam polarimeters images."""

import abc
import numpy as np
from dataclasses import dataclass
from astropy.table import Table
from scipy.spatial import cKDTree
from astropy import units

from ..logger import logger
from ..math.physical import QFloat


__all__ = ['estimate_dxdy', 'match_pairs',
           'quarterwave_model', 'halfwave_model']


def _compute_theta(q, u):
    """Compute theta using Q and U, considering quadrants and max 180 value."""
    # numpy arctan2 already looks for quadrants and is defined in [-pi, pi]
    theta = np.degrees(0.5*np.arctan2(u, q))
    # do not allow negative values
    if theta < 0:
        theta += 180
    return theta*units.degree


def estimate_dxdy(x, y, steps=[100, 30, 5, 3], bins=30, dist_limit=100):
    """Estimate the displacement between the two beams.

    To compute the displacement between the ordinary and extraordinary
    beams, this function computes the most common distances between the
    sources in image, using clipped histograms around the peak.
    """
    def _find_max(d):
        dx = 0
        for lim in (np.max(d), *steps):
            lo, hi = (dx-lim, dx+lim)
            lo, hi = (lo, hi) if (lo < hi) else (hi, lo)
            histx = np.histogram(d, bins=bins, range=[lo, hi])
            mx = np.argmax(histx[0])
            dx = (histx[1][mx]+histx[1][mx+1])/2
        return dx

    # take all combinations
    comb = np.array(np.meshgrid(np.arange(len(x)),
                                np.arange(len(x)))).T.reshape(-1, 2)
    # filter only y[j] > y[i]
    filt = y[comb[:, 1]] > y[comb[:, 0]]
    comb = comb[np.where(filt)]

    # compute the distances
    dx = x[comb[:, 0]] - x[comb[:, 1]]
    dy = y[comb[:, 0]] - y[comb[:, 1]]

    # filter by distance
    filt = (np.abs(dx) <= dist_limit) & (np.abs(dy) <= dist_limit)
    dx = dx[np.where(filt)]
    dy = dy[np.where(filt)]

    logger.debug(f"Determining the best dx,dy with {len(dx)} combinations.")

    return (_find_max(dx), _find_max(dy))


def match_pairs(x, y, dx, dy, tolerance=1.0):
    """Match the pairs of ordinary/extraordinary points (x, y)."""
    kd = cKDTree(list(zip(x, y)))

    px = np.array(x-dx)
    py = np.array(y-dy)

    d, ind = kd.query(list(zip(px, py)), k=1, distance_upper_bound=tolerance,
                      workers=-1)

    o = np.arange(len(x))[np.where(d <= tolerance)]
    e = np.array(ind[np.where(d <= tolerance)])
    result = Table()
    result['o'] = o
    result['e'] = e

    return result.as_array()


def quarterwave_model(psi, q, u, v, zero=0):
    """Compute polarimetry z(psi) model for quarter wavelength retarder.

    Z(psi) = Q*cos(2psi)**2 + U*sin(2psi)*cos(2psi) - V*sin(2psi)

    Parameters
    ----------
    psi: array_like
        Array of retarder positions in degrees.
    q, u, v: float
        Stokes parameters
    zero: float
        Zero position of the retarder in degrees.

    Return
    ------
    z: array_like
        Array of polarimetry values.
    """
    if zero is not None:
        psi = psi+zero  # avoid inplace modification
    psi = np.radians(psi)
    psi2 = 2*psi
    zi = q*(np.cos(psi2)**2) + u*np.sin(psi2)*np.cos(psi2) - v*np.sin(psi2)
    return zi


def halfwave_model(psi, q, u, zero=None):
    """Compute polarimetry z(psi) model for half wavelength retarder.

    Z(psi) = Q*cos(4*psi) + U*sin(4*psi)

    Parameters
    ----------
    psi: array_like
        Array of retarder positions in degrees.
    q, u: float
        Stokes parameters
    zero: float (optional)
        Zero angle of the retarder position.

    Return
    ------
    z: array_like
        Array of polarimetry values.
    """
    if zero is not None:
        psi = psi+zero  # avoid inplace modification
    psi = np.radians(psi)
    zi = q*np.cos(4*psi) + u*np.sin(4*psi)
    return zi


@dataclass
class _DualBeamPolarimetry(abc.ABC):
    """Base class for polarimetry computation."""

    retarder: str  # 'quarterwave' or 'halfwave'
    k: float = None  # global normalization constant
    zero: float = None  # zero position of the retarder
    compute_zero: bool = False  # compute the zero position
    compute_k: bool = False  # compute the normalization constant
    min_snr: float = None  # minimum signal-to-noise ratio
    psi_deviation: float = 0.1  # max deviation of retarder position

    def __post_init__(self):
        if self.retarder not in ['quarterwave', 'halfwave']:
            raise ValueError(f"Retarder {self.retarder} unknown.")
        if isinstance(self.zero, (QFloat, units.Quantity)):
            self.zero = self.zero.to(units.degree).value
        if self.k is not None and self.compute_k:
            raise ValueError('k and compute_k cannot be used together.')
        if self.zero is not None and self.compute_zero:
            raise ValueError('zero and compute_zero cannot be used together.')

        if self.compute_zero and self.retarder == 'halfwave':
            raise ValueError('Half-wave retarder cannot compute zero.')

        # number of positions per cicle
        self._n_pos = 8 if self.retarder == 'quarterwave' else 4

    def _check_positions(self, psi):
        """Check if positions are in the correct order."""
        devs = np.abs(psi/22.5 - np.round(psi/22.5, 0))*22.5
        if np.any(devs > self.psi_deviation):
            raise ValueError("Retarder positions must be multiple of 22.5 deg")

    def _calc_zi(self, f_ord, f_ext, k):
        """Compute zi from ordinary and extraordinary fluxes."""
        f_ord = np.array(f_ord, dtype=float)
        f_ext = np.array(f_ext, dtype=float)
        return (f_ord - f_ext*k)/(f_ord + f_ext*k)

    def _estimate_normalize_half(self, psi, f_ord, f_ext):
        """Estimate the normalization factor for halfwave retarder."""
        pos_in_cycle = np.mod(np.floor_divide(psi, 22.5), self._n_pos)
        pos_in_cycle = pos_in_cycle.astype(int)
        ford_mean = np.full(self._n_pos, np.nan)
        fext_mean = np.full(self._n_pos, np.nan)

        for i in range(self._n_pos):
            ford_mean[i] = np.mean(f_ord[pos_in_cycle == i])
            fext_mean[i] = np.mean(f_ext[pos_in_cycle == i])

        if np.any(np.isnan(ford_mean)) or np.any(np.isnan(fext_mean)):
            raise ValueError('Could not estimate the normalization factor.')

        return np.sum(f_ord)/np.sum(f_ext)

    def _estimate_normalize_quarter(self, q):
        """Estimate the normalization factor for quarterwave retarder."""
        return (1+0.5*q)/(1-0.5*q)


@dataclass
class SLSDualBeamPolarimetry(_DualBeamPolarimetry):
    """Polarimetry computation for Stokes Least Squares algorithm.

    This method is describe in [1]_ and consists in fitting the data using
    theoretical models. This method has the advantage of not need particular
    sets of retarder positions and can handle missing points.

    Parameters
    ----------
    retarder: str
        Retarder type. Must be 'quarterwave' or 'halfwave'.
    k: float (optional)
        Normalization factor. If None, it is estimated from the data.
        Default is None.
    zero: float (optional)
        Zero position of the retarder in degrees. If None, it is estimated
        from the data. Defult is None.
    compute_zero: bool (optional)
        Fit zero position using the data. Default is False. Conflicts with
        ``zero`` argument.
    compute_k: bool (optional)
        Fit the normalization factor using the data. Default is False.
        Conflicts with ``k`` argument.
    min_snr: float (optional)
        Minimum signal-to-noise ratio. Points with lower SNR will be discarded.
        Default is None.
    psi_deviation: float (optional)
        Maximum deviation of the psi position from the sequence multiple
        of 22.5 degrees. Default is 0.1 degrees.

    Notes
    -----
    - The model fitting is performed by `~scipy.optimize.cure_fit` function,
      using the Trust Region Reflective ``trf`` method.
    - If ``k`` or ``zero`` arguments are passed, they won't be computed.
      Instead, the passed values will be used.
    - If ``compute_zero`` or ``compute_k`` are True, the values for these
      constants will be estimated from the data.

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2019PASP..131b4501N
    """


@dataclass
class PCCDDualBealPlarimetry(_DualBeamPolarimetry):
    """Polarimetry computation using PCCDPACK algorithms.

    PCCDPACK algorithms are described by [1]_ and [2]_.

    Parameters
    ----------
    retarder: str
        Retarder type. Must be 'quarterwave' or 'halfwave'.
    k: float (optional)
        Normalization factor. If None, it is estimated from the data.
    zero: float (optional)
        Zero position of the retarder in degrees. If None, it is estimated
        from the data.
    compute_zero: bool (optional)
        Fit zero position using the data.
    compute_k: bool (optional)
        Fit the normalization factor using the data.

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/1984PASP...96..383M
    .. [2] https://ui.adsabs.harvard.edu/abs/1998A&A...335..979R
    """

    def _half_compute(self, psi, zi):
        """Compute stokes parameters for halfwave retarder."""
        n = len(psi)
        q = (2.0/n) * np.nansum(zi*np.cos(4*np.radians(psi)))
        u = (2.0/n) * np.nansum(zi*np.sin(4*np.radians(psi)))
        return q, u

    def _quarter_compute(self, psi, zi):
        """Compute stokes parameters for quarterwave retarder."""
        # FIXME: the sums is from 0 to 7, 1-cycle only
        psi = np.radians(psi)
        q = 1.0/3 * np.nansum(zi*np.square(np.cos(2*psi)))
        u = np.nansum(zi*np.sin(2*psi)*np.cos(2*psi))
        v = -1.0/4 * np.nansum(zi*np.sin(2*psi))
        return q, u, v
