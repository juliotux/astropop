# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.modeling import custom_model
from scipy.spatial import cKDTree
from astropy.table import Table

from ..logger import logger


__all__ = ['estimate_dxdy', 'match_pairs',
           'HalfWaveModel', 'QuarterWaveModel']


def estimate_dxdy(x, y, steps=[100, 30, 5, 3], bins=30, dist_limit=100,
                  logger=logger):
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

    logger.debug("Determining the best dx,dy with {} combinations."
                 .format(len(dx)))

    return (_find_max(dx), _find_max(dy))


def match_pairs(x, y, dx, dy, tolerance=1.0, logger=logger):
    """Match the pairs of ordinary/extraordinary points (x, y)."""
    kd = cKDTree(list(zip(x, y)))

    px = np.array(x-dx)
    py = np.array(y-dy)

    d, ind = kd.query(list(zip(px, py)), k=1, distance_upper_bound=tolerance,
                      n_jobs=-1)

    o = np.arange(len(x))[np.where(d <= tolerance)]
    e = np.array(ind[np.where(d <= tolerance)])
    result = Table()
    result['o'] = o
    result['e'] = e

    return result.as_array()


def quarter(psi, q=1.0, u=1.0, v=1.0):
    '''Polarimetry z(psi) model for quarter wavelenght retarder.

    Z= Q*cos(2psi)**2 + U*sin(2psi)*cos(2psi) - V*sin(2psi)'''
    psi2 = 2*psi
    z = q*(np.cos(psi2)**2) + u*np.sin(psi)*np.cos(psi2) - v*np.sin(psi2)
    return z


def quarter_deriv(psi, q=1.0, u=1.0, v=1.0):
    x = 2*psi
    dq = np.cos(x)**2
    du = 0.5*np.sin(2*x)
    dv = -np.sin(2*x)
    return (dq, du, dv)


def half(psi, q=1.0, u=1.0):
    '''Polarimetry z(psi) model for half wavelenght retarder.

    Z(I)= Q*cos(4psi(I)) + U*sin(4psi(I))'''
    return q*np.cos(4*psi) + u*np.sin(4*psi)


def half_deriv(psi, q=1.0, u=1.0):
    return (np.cos(4*psi), np.sin(4*psi))


HalfWaveModel = custom_model(half, fit_deriv=half_deriv)
QuarterWaveModel = custom_model(quarter, fit_deriv=quarter_deriv)
