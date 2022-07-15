# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Compute polarimetry of dual beam polarimeters images."""

import abc
import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import Table
from scipy.spatial import cKDTree
from astropy.modeling import Fittable1DModel, Parameter
from astropy import units

from ..logger import logger


# TODO: Reimplement normalization
# TODO: Implement quarter-wave for MBR84

__all__ = ['estimate_dxdy', 'match_pairs',
           'MBR84DualBeamPolarimetry', 'SLSDualBeamPolarimetry',
           'HalfWaveModel', 'QuarterWaveModel']


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
                      n_jobs=-1)

    o = np.arange(len(x))[np.where(d <= tolerance)]
    e = np.array(ind[np.where(d <= tolerance)])
    result = Table()
    result['o'] = o
    result['e'] = e

    return result.as_array()


class HalfWaveModel(Fittable1DModel):
    """Polarimetry z(psi) model for half wavelenght retarder.

    Z(I)= Q*cos(4psi(I)) + U*sin(4psi(I))
    psi in degrees.
    """

    q = Parameter(default=0)
    u = Parameter(default=0)

    @property
    def p(self):
        return np.sqrt(self.q**2 + self.u**2)

    @property
    def theta(self):
        return _compute_theta(self.q, self.u)

    @staticmethod
    def evaluate(psi, q, u):
        return q*np.cos(4*psi) + u*np.sin(4*psi)

    @staticmethod
    def fit_deriv(psi, q, u):
        dq = np.cos(4*psi)
        du = np.sin(4*psi)
        return (dq, du)


class QuarterWaveModel(Fittable1DModel):
    """Polarimetry z(psi) model for quarter wavelength retarder.

    Z= Q*cos(2psi)**2 + U*sin(2psi)*cos(2psi) - V*sin(2psi)
    psi in degrees.
    """

    q = Parameter(default=0)
    u = Parameter(default=0)
    v = Parameter(default=0)
    zero = Parameter(default=0, unit=units.degree, fixed=True)

    @property
    def p(self):
        return np.sqrt(self.q**2 + self.u**2)

    @property
    def theta(self):
        return _compute_theta(self.q, self.u)

    @staticmethod
    def evaluate(x, q, u, v, zero):
        psi = np.radians(x+zero)
        psi2 = 2*psi
        r = q*(np.cos(psi2)**2) + u*np.sin(psi2)*np.cos(psi2) - v*np.sin(psi2)
        return r

    @staticmethod
    def fit_deriv(x, q, u, v, zero):
        psi = np.radians(x+zero)
        dq = np.cos(psi)**2
        du = 0.5*np.sin(2*psi)
        dv = -np.sin(2*psi)
        return (dq, du, dv)

    @property
    def input_units(self):
        return {self.inputs[0]: self.zero.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {'q': outputs_unit[self.outputs[0]],
                'u': outputs_unit[self.outputs[0]],
                'v': outputs_unit[self.outputs[0]]}


class DualBeamPolarimetryBase(abc.ABC):
    """Base class for polarimetry computation."""

    def __init__(self, retarder, normalize=True, positions=None, min_snr=None,
                 filter_negative=True, global_k=None):
        self._retarder = retarder
        self._min_snr = min_snr
        self._normalize = normalize
        self._filter_negative = filter_negative
        self._global_k = global_k
        self._logger = logger

    @property
    def retarder(self):
        """Retarder used for polarimetry computations."""
        return self._retarder

    @property
    def min_snr(self):
        """Minimal SNR which will be returned. Bellow, return NaN."""
        return self._min_snr

    @property
    def normalize(self):
        """Normalize of beams enabled."""
        return self._normalize

    @property
    def filter_negative(self):
        """Negative fluxes filtered."""
        return self._filter_negative

    @property
    def global_k(self):
        """Return the k constant value for all computations."""
        return self._global_k

    @abc.abstractmethod
    def compute(self, psi, ford, fext, ford_err=None, fext_err=None,
                logger=None):
        """Compute the polarimetry."""

    @abc.abstractmethod
    def estimate_normalize(self, psi, ford, fext):
        """Estimate the normalization constant."""

    def calc_z(self, psi, ford, fext, ford_err=None, fext_err=None):
        """Calculate Z value using ford and fext."""
        # clean problematic sources (bad sky subtraction, low snr)
        self._filter_neg(ford, fext)  # inplace

        if self.normalize:
            if self.global_k is not None:
                k = self.global_k
            else:
                k = self.estimate_normalize(psi, ford, fext)
        else:
            k = 1

        z = (ford-(fext*k))/(ford+(fext*k))

        if ford_err is None or fext_err is None:
            z_err = None
        else:
            # Assuming individual z errors from propagation
            ford_err = np.array(ford_err)
            fext_err = np.array(fext_err)
            oi = 2*ford/((ford+fext)**2)
            ei = -2*fext/((ford+fext)**2)
            z_err = np.sqrt((oi**2)*(ford_err**2) + ((ei**2)*(fext_err**2)))
        return z, z_err

    def _filter_neg(self, ford, fext):
        """Filter the negative values. Inplace."""
        if self.filter_negative:
            filt = (ford < 0) | (fext < 0)
            w = np.where(~filt)
            ford[w] = np.nan
            fext[w] = np.nan


class SLSDualBeamPolarimetry(DualBeamPolarimetryBase):
    """Compute polarimetry using SLS method.

    The Stokes Least Squares (SLS) method consists in fit the relative
    difference between the ordinary and extraordinary beams to defined
    equations, that depends on what retarder is being used.
    The fitting is performed using Levenberg–Marquardt algorith. For half-wave
    retarders, we use:

    Z(I)= Q*cos(4psi(I)) + U*sin(4psi(I))

    For quarter wave retarders:

    Z= Q*cos(2psi)**2 + U*sin(2psi)*cos(2psi) - V*sin(2psi)

    More details can be found Campagnolo 2019 (ads: 2019PASP..131b4501N)
    """

    def __init__(self, retarder, normalize=True, positions=None, min_snr=None,
                 filter_negative=True, global_k=None, **kwargs):
        super(SLSDualBeamPolarimetry, self).__init__(retarder, normalize,
                                                     positions, min_snr,
                                                     filter_negative, global_k)
        if self.retarder == 'half':
            self._model = HalfWaveModel
        elif self.retarder == 'quarter':
            self._model = QuarterWaveModel
        elif self.retarder == 'other':
            raise NotImplementedError('Generic retarder not implemented')
        else:
            raise ValueError(f'Retarder {self.retarder} not recognized.')

    def compute(self, psi, ford, fext, ford_err=None, fext_err=None,
                logger=None):
        """Compute the polarimetry.

        Parameters
        ----------
        psi : array_like
            Retarder positions in degrees
        ford, fext : array_like
            Fluxes of ordinary (ford) and extraordinary (fext) beams.
        ford_err, fext_err : array_like
            Statistical errors of ordinary and extraordinary fluxes.
        logger : `logging.Logger`
            Python logger of the function.

        Notes
        -----
        * `psi`, `ford` and `fext` must match the dimensions.

        * If each data have just one dimension, it will be considered
          a single star.

        * If each data have two dimensions, it will be considered multiple
          stars, where each line representes one star.
        """
        logger = logger or self.logger

        self._filter_neg(ford, fext)  # inplace
        z, z_err = self.calc_z(psi, ford, fext, ford_err, fext_err)

        n_stars = len(z)
        logger.info(f'Computing polarimetry for {n_stars} stars.')

        # Variables to store the results
        res = Table()
        res['z'] = z
        res['z_err'] = z_err
        for i in ('q', 'u'):
            res[i] = np.zeros(n_stars, dtype='f8')
            res[i+"_err"] = np.zeros(n_stars, dtype='f8')
            res[i].fill(np.nan)  # fill with nan to be safer
            res[i+"_err"].fill(np.nan)
        if self.retarder != 'half':
            res['v'] = np.zeros(n_stars, dtype='f8')
            res['v_err'] = np.zeros(n_stars, dtype='f8')
            res['v'].fill(np.nan)
            res['v_err'].fill(np.nan)

        for i in range(n_stars):
            fitter = LevMarLSQFitter()
            model = self._model()
            if z_err is not None:
                m_fit = fitter(model, psi[i], z[i], weights=1/z_err[i])
            else:
                m_fit = fitter(model, psi[i], z[i])
            info = fitter.fit_info
            for n, v, err in zip(m_fit.param_names, m_fit.parameters,
                                 np.sqrt(np.diag(info['param_cov']))):
                res[n][i] = v
                res[n+"_err"] = err

        res['p'] = np.hipot(res['q'], res['u'])
        res['p_err'] = np.sqrt(((res['q']/res['p'])**2)*(res['q_err']**2) +
                               ((res['u']/res['p'])**2)*(res['u_err']**2))
        res['theta'] = compute_theta(res['q'], res['u'])
        res['theta_err'] = 28.65*res['p_err']/res['p']

        return res


class MBR84DualBeamPolarimetry(DualBeamPolarimetryBase):
    """Compute polarimetry using MBR84 method.

    Method Described by Magalhaes et al 1984 (ads string: 1984PASP...96..383M)
    """

    def __init__(self, retarder, normalize=True, positions=None, min_snr=None,
                 filter_negative=True, global_k=None, **kwargs):
        super(SLSDualBeamPolarimetry, self).__init__(retarder, normalize,
                                                     positions, min_snr,
                                                     filter_negative, global_k)

    def compute(self, psi, ford, fext, ford_err=None, fext_err=None,
                logger=None):
        """Compute the polarimetry.

        Parameters
        ----------
        psi : array_like
            Retarder positions in degrees
        ford, fext : array_like
            Fluxes of ordinary (ford) and extraordinary (fext) beams.
        ford_err, fext_err : array_like
            Statistical errors of ordinary and extraordinary fluxes.
        logger : `logging.Logger`
            Python logger of the function.

        Notes
        -----
        * `psi`, `ford` and `fext` must match the dimensions.

        * If each data have just one dimension, it will be considered
          a single star.

        * If each data have two dimensions, it will be considered multiple
          stars, where each line representes one star.
        """
        logger = logger or self.logger
        self._filter_neg(ford, fext)  # inplace

        result = Table()
        z, z_err = self.calc_z(psi, ford, fext, ford_err, fext_err)
        logger.info(f'Computing polarimetry for {len(z)} stars.')

        if self.retarder == 'half':
            n = len(z)
            q = (2.0/n) * np.nansum(z*np.cos(4*psi))
            u = (2.0/n) * np.nansum(z*np.sin(4*psi))
            p = np.sqrt(q**2 + u**2)

            a = 2.0/n
            b = np.sqrt(1.0/(n-2))
            err = a*np.nansum(z**2)
            err = err - p**2
            err = b*np.sqrt(err)

            result['q'] = q
            result['q_err'] = err
            result['u'] = u
            result['u_err'] = err
            result['p'] = p
            result['p_err'] = err
            result['theta'] = compute_theta(q, u)
            result['theta_err'] = 28.65*err/p
        else:
            raise ValueError(f'Retarder {self.retarder} not supported')

        return result
