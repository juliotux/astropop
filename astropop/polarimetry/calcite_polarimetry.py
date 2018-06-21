# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from scipy.spatial import cKDTree
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import Table

from .polarimetry_models import HalfWaveModel, QuarterWaveModel
from ..logger import logger


def estimate_dxdy(x, y, steps=[100, 30, 5, 3], bins=30):
    def _find_max(d):
        dx = 0
        for lim in (np.max(d), *steps):
            lo, hi = (dx-lim, dx+lim)
            lo, hi = (lo, hi) if (lo < hi) else (hi, lo)
            histx = np.histogram(d, bins=bins, range=[lo, hi])
            mx = np.argmax(histx[0])
            dx = (histx[1][mx]+histx[1][mx+1])/2
        return dx

    dya = []
    dxa = []

    for i in range(len(x)):
        for j in range(len(x)):
            if y[i] < y[j]:
                dya.append(y[i] - y[j])
                dxa.append(x[i] - x[j])

    return (_find_max(dxa), _find_max(dya))


def match_pairs(x, y, dx, dy, tolerance=1.0):
    """Match the pairs of ordinary/extraordinary points (x, y)."""
    dt = np.dtype([('o', int), ('e', int)])
    results = np.zeros(len(x), dtype=dt)
    npairs = 0

    p = list(zip(x, y))
    kd = cKDTree(p)

    for i in range(len(p)):
        px = p[i][0]-dx
        py = p[i][1]-dy
        d, j = kd.query((px, py), k=1, eps=tolerance,
                        distance_upper_bound=tolerance, n_jobs=-1)
        if d <= tolerance:
            results[npairs]['o'] = i
            results[npairs]['e'] = j
            npairs = npairs+1
            kd = cKDTree(p)

    return results[:npairs]


def estimate_normalize(o, e, positions, n_consecutive):
    """Estimate the normalization of a given set of data.
    """
    data_o = [[]]*n_consecutive
    data_e = [[]]*n_consecutive

    # First, we separate the data in the positions, relative to consecutive
    for i, oi, ei in zip(positions, o, e):
        index = int(i/n_consecutive)
        data_o[index].append(oi)
        data_e[index].append(ei)

    # check if all positions have a value
    for i in data_o:
        if i == []:
            logger.warn('Could not calculate polarimetry normalization. '
                        'Not all needed positions are available. Using k=1.')
            return 1

    # Now we use as each consecutive value the mean of the values in each index
    for i in range(n_consecutive):
        data_o[i] = np.nanmean(data_o[i])
        data_e[i] = np.nanmean(data_e[i])

    # Now, assuming the k will multiply e
    k = np.sum(data_o)/np.sum(data_e)
    logger.debug('Polarimetry normalization estimated as k={}'.format(k))
    return k


def compute_theta(q, u):
    '''Giving q and u, compute theta'''
    # numpy arctan2 already looks for quadrants and is defined in [-pi, pi]
    theta = np.degrees(0.5*np.arctan2(u, q))
    # do not allow negative values
    if theta < 0:
        theta += 180
    return theta


def _calculate_polarimetry_parameters(z, psi, retarder='half', z_err=None):
    """Calculate the polarimetry directly using z.
    psi in degrees
    """
    result = {}
    if z_err is None:
        result['z'] = {'value': z,
                       'sigma': np.array([np.nan]*len(z))}
    else:
        result['z'] = {'value': z, 'sigma': z_err}

    if retarder == 'half':
        model = HalfWaveModel()
    elif retarder == 'quarter':
        model = QuarterWaveModel()
    else:
        raise ValueError('retarder {} not supported.'.format(retarder))

    psi = np.radians(psi)

    fitter = LevMarLSQFitter()
    if z_err is None:
        m_fitted = fitter(model, psi, z)
    else:
        m_fitted = fitter(model, psi, z, weights=1/z_err)
    info = fitter.fit_info
    # The errors of parameters are assumed to be the sqrt of the diagonal of
    # the covariance matrix
    for i, j, k in zip(m_fitted.param_names, m_fitted.parameters,
                       np.sqrt(np.diag(info['param_cov']))):
        result[i] = {'value': j, 'sigma': k}

    if z_err is not None:
        result['sigma_theor'] = np.sqrt(np.sum(np.square(z_err))/len(z))
    else:
        result['sigma_theor'] = np.nan

    q, u = result['q']['value'], result['u']['value']
    q_err, u_err = result['q']['sigma'], result['u']['sigma']

    p = np.sqrt(q**2 + u**2)
    p_err = np.sqrt(((q/p)**2)*(q_err**2) + ((u/p)**2)*(u_err**2))
    result['p'] = {'value': p, 'sigma': p_err}

    theta = compute_theta(q, u)
    result['theta'] = {'value': theta, 'sigma': 28.65*p_err/p}

    return result


def calculate_polarimetry(o, e, psi, retarder='half', o_err=None, e_err=None,
                          normalize=True, positions=None, min_snr=None,
                          filter_negative=True):
    """Calculate the polarimetry."""

    if retarder == 'half':
        ncons = 4
    elif retarder == 'quarter':
        ncons = 8
    else:
        raise ValueError('retarder {} not supported.'.format(retarder))

    o = np.array(o)
    e = np.array(e)

    # clean problematic sources (bad sky subtraction, low snr)
    if filter_negative and (np.array(o <= 0).any() or np.array(e <= 0).any()):
        o_neg = np.where(o < 0)
        e_neg = np.where(e < 0)
        o[o_neg] = np.nan
        e[e_neg] = np.nan

    if normalize and positions is not None:
        k = estimate_normalize(o, e, positions, ncons)
        z = (o-(e*k))/(o+(e*k))
    else:
        z = (o-e)/(o+e)

    # To fit pccdpack, we had to invert z
    z = -z

    if o_err is None or e_err is None:
        z_erro = None
    else:
        # Assuming individual z errors from propagation
        o_err = np.array(o_err)
        e_err = np.array(e_err)
        oi = 2*o/((o+e)**2)
        ei = -2*e/((o+e)**2)
        z_erro = np.sqrt((oi**2)*(o_err**2) + ((ei**2)*(e_err**2)))

    flux = np.sum(o)+np.sum(e)
    flux_err = np.sqrt(np.sum(o_err)**2 + np.sum(e_err)**2)

    def _return_empty():
        if retarder == 'half':
            keys = ['q', 'u']
        elif retarder == 'quarter':
            keys = ['q', 'u', 'v']
        dic = {}
        for i in keys + ['p', 'theta']:
            dic[i] = {'value': np.nan, 'sigma': np.nan}
        dic['z'] = {'value': z, 'sigma': z_erro}
        dic['sigma_theor'] = np.nan
        dic['flux'] = {'value': flux,
                       'sigma': flux_err}

        return dic

    if min_snr is not None and o_err is not None and e_err is not None:
        snr = flux/flux_err
        if snr < min_snr:
            logger.debug('Star with SNR={} eliminated.'.format(snr))
            return _return_empty()

    try:
        result = _calculate_polarimetry_parameters(z, psi, retarder=retarder,
                                                   z_err=z_erro)
    except Exception as e:
        return _return_empty()
    result['flux'] = {'value': flux,
                      'sigma': flux_err}

    return result
