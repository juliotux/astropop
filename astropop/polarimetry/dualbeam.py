# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Compute polarimetry of dual beam polarimeters images."""

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree
from astropy import units

from ..logger import logger


# TODO: Reimplement normalization
# TODO: Implement quarter-wave for MBR84
# TODO: for future, change scipy fitting to astropy.modeling


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
