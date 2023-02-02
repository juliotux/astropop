# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Compute polarimetry of dual beam polarimeters images."""

import abc
import numpy as np
from dataclasses import dataclass
from astropy.table import Table
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from astropy import units
from functools import partial

from ..logger import logger
from ..math.physical import QFloat


__all__ = ['estimate_dxdy', 'match_pairs',
           'quarterwave_model', 'halfwave_model']


def _compute_theta(q, u):
    """Compute theta using Q and U, considering quadrants and max 180 value."""
    # numpy arctan2 already looks for quadrants and is defined in [-pi, pi]
    theta = np.degrees(0.5*np.arctan2(u, q))
    if not hasattr(theta, 'unit'):
        theta = theta*units.degree
    # do not allow negative values
    if theta < 0*units.degree:
        theta += 180*units.degree
    return theta


def estimate_dxdy(x, y, steps=[100, 30, 5, 3], bins=30, dist_limit=100):
    """Estimate the displacement between the two beams.

    To compute the displacement between the ordinary and extraordinary
    beams, this function computes the most common distances between the
    sources in image, using clipped histograms around the peak.

    Parameters
    ----------
    x, y: array_like
        Arrays of x and y positions of the sources.
    steps: list of int
        List of steps to use in the histogram.
    bins: int
        Number of bins to use in the histogram.
    dist_limit: float
        Maximum distance between the pairs of sources to consider.

    Return
    ------
    dx, dy: float
        Displacement between the ordinary and extraordinary beams.
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
    """Match the pairs of ordinary/extraordinary points (x, y).

    This function matches the pairs of ordinary and extraordinary points
    (x, y) using the displacement (dx, dy) between the two beams.

    Parameters
    ----------
    x, y: array_like
        Arrays of x and y positions of the sources.
    dx, dy: float
        Displacement between the ordinary and extraordinary beams.
    tolerance: float
        Tolerance for the matching.

    Return
    ------
    `numpy.ndarray`:
        Array of the indexes of the matched pairs. Column 'o' contains the
        indexes of the ordinary points, column 'e' the indexes of the
        extraordinary points.
    """
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
class StokesParameters:
    """Store the Stokes parameters results from dual beam polarimeters.

    Parameters
    ----------
    q, u, v: `~astropop.math.QFloat`
        Stokes parameters.
    retarder: str
        'quarterwave' or 'halfwave'
    k: float
        Normalization constant used to compute zi.
    zero: float
        Zero position of the retarder in degrees.
    zi: `~astropop.math.QFloat`
        Relative difference of fluxes between ordinary and extraordinary beams.
    psi: array_like
        Array of retarder positions in degrees.
    flux: `~astropop.math.QFloat` (optional)
        Sum of ordinary and extraordinary counts in each retarder position.
    """

    retarder: str  # 'quarterwave' or 'halfwave'
    q: QFloat
    u: QFloat
    v: QFloat = None
    k: float = 1.0
    zero: QFloat = 0.0
    flux: QFloat = None
    zi: QFloat = None
    psi: QFloat = None

    def __post_init__(self):
        # Check if all variables are correct
        self.retarder = str(self.retarder)
        if self.retarder not in ('halfwave', 'quarterwave'):
            raise ValueError('retarder must be halfwave or quarterwave')

        self.q = QFloat(self.q)
        self.u = QFloat(self.u)
        if self.v is not None:
            self.v = QFloat(self.v)

        if self.k is not None:
            self.k = float(self.k)
        if self.zero is not None:
            self.zero = QFloat(self.zero)
            if self.zero.unit == units.dimensionless_unscaled:
                self.zero.unit = 'deg'

        if self.zi is not None:
            self.zi = QFloat(self.zi)
        if self.psi is not None:
            self.psi = QFloat(self.psi)
            if self.psi.unit == units.dimensionless_unscaled:
                self.psi.unit = 'deg'

        if self.psi is not None and self.zero is not None:
            if self.psi.unit != self.zero.unit:
                raise ValueError('Psi and Zero have different units.')

        # flux, zi and psi must have the same dimensions if exists
        length = None
        for i in (self.psi, self.zi, self.flux):
            if i is not None:
                if length is None:
                    length = len(i)
                elif len(i) != length:
                    raise ValueError('psi, zi and flux must have the same '
                                     'dimensions')

    @property
    def p(self):
        """Linear polarization level."""
        return np.hypot(self.q, self.u)

    @property
    def theta(self):
        """Angle between Stokes parameters."""
        theta = _compute_theta(self.q, self.u).nominal
        err = 28.6*self.p.std_dev/self.p.nominal
        return QFloat(theta, err, 'deg')

    @property
    def rms(self):
        """Root mean square of the fitting."""
        if self.zi is None or self.psi is None:
            raise ValueError('StokesParameters without zi and psi data has no '
                             'fitting rms')
        return np.std(self.zi-self.model(self.psi))

    @property
    def model(self):
        """Callable model of the modulation (zi)."""
        if self.retarder == 'quarterwave':
            model = partial(quarterwave_model, q=self.q, u=self.u, v=self.v,
                            zero=self.zero)
        else:
            model = partial(halfwave_model, q=self.q, u=self.u, zero=self.zero)
        return model

    @property
    def theor_sigma(self):
        """Theoretical sigma of the polarization level.

        It is expressed as:
        theor_sigma = K*1/sqrt(sum(flux_i^2/sigma_flux_i^2))
        where K is 1 for halfwave retarders and sqrt2 for quarterwave.
        """
        if self.flux is None:
            raise ValueError('The theoretical sigma is only available when '
                             'fluxes are present.')
        ratio = self.flux.nominal/self.flux.std_dev
        snr = np.sqrt(np.sum(np.square(ratio)))
        if self.retarder == 'halfwave':
            sigma = np.sqrt(2)/snr
            return {'q': sigma, 'u': sigma, 'p': sigma}
        if self.retarder == 'quarterwave':
            sigma_q = 1/(np.sqrt(0.396)*snr)
            sigma_u = 1/(np.sqrt(0.1464)*snr)
            sigma_p = np.sqrt((self.q.nominal*sigma_q)**2 +
                              (self.u.nominal*sigma_u)**2)/self.p.nominal
            return {'q': sigma_q,
                    'u': sigma_u,
                    'v': 1/(np.sqrt(0.4571)*snr),
                    'p': sigma_p}


@dataclass
class _DualBeamPolarimetry(abc.ABC):
    """Base class for polarimetry computation."""

    retarder: str  # 'quarterwave' or 'halfwave'
    k: float = None  # global normalization constant
    zero: float = None  # zero position of the retarder. If not set, computed.
    compute_k: bool = False  # compute the normalization constant
    min_snr: float = None  # minimum signal-to-noise ratio
    psi_deviation: float = 0.1  # max deviation of retarder position
    iter_tolerance: float = 1e-5  # maximum tolerance on the iterative fitting
    max_iters: int = 100  # maximum number of iterations

    def __post_init__(self):
        if self.retarder not in ['quarterwave', 'halfwave']:
            raise ValueError(f"Retarder {self.retarder} unknown.")
        if isinstance(self.zero, (QFloat, units.Quantity)):
            self.zero = self.zero.to(units.degree).value
        if self.k is not None and self.compute_k:
            raise ValueError('k and compute_k cannot be used together.')
        if not self.compute_k:
            logger.info('Normalization disabled.')

        # number of positions per cicle
        self._n_pos = 8 if self.retarder == 'quarterwave' else 4

    def _check_positions(self, psi):
        """Check if positions are in the correct order."""
        # Only positions multiple of 22.5 degrees are allowed
        devs = np.abs(psi/22.5 - np.round(psi/22.5, 0))*22.5
        if np.any(devs > self.psi_deviation):
            raise ValueError("Retarder positions must be multiple of 22.5 deg")

    def _calc_zi(self, f_ord, f_ext, k):
        """Compute zi from ordinary and extraordinary fluxes."""
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

    def _estimate_normalize_quarter(self, psi, f_ord, f_ext, q):
        """Estimate the normalization factor for quarterwave retarder."""
        ratio = self._estimate_normalize_half(psi, f_ord, f_ext)
        q_norm = (1+0.5*q)/(1-0.5*q)
        if (ratio < 1 and q < 0) or (ratio > 1 and q > 0):
            q_norm = 1/q_norm
        return ratio*q_norm

    def _half_compute(self, psi, f_ord, f_ext):
        """Compute the Stokes params for halfwave retarder."""
        fluxes = f_ord + f_ext
        # estimate normalization factor
        if self.k is not None:
            k = self.k
        elif self.compute_k:
            k = self._estimate_normalize_half(psi, f_ord, f_ext)
        else:
            k = 1.0

        # compute Stokes params
        zi = self._calc_zi(f_ord, f_ext, k)
        q, u = self._half_fit(psi, zi)
        if self.zero is not None:
            zero = QFloat(self.zero, unit='deg')
        else:
            zero = None
        psi = QFloat(psi, unit='deg')
        return StokesParameters('halfwave', q=q, u=u, v=None, k=QFloat(k),
                                zero=zero, psi=psi, zi=zi, flux=fluxes)

    def _quarter_compute(self, psi, f_ord, f_ext):
        """Compute the Stokes params for quarterwave retarder."""
        fluxes = f_ord + f_ext
        # bypass normalization
        if not self.compute_k:
            k = self.k or 1.0
            zi = self._calc_zi(f_ord, f_ext, k)
            params = self._quarter_fit(psi, zi)
        else:
            # iterate over k until the diference previous and current values
            # is smaller than the tolerance
            k = 1.0
            previous = {'q': 0, 'u': 0, 'v': 0}
            converged = False
            for i in range(self.max_iters):
                # compute Stokes params, dict(q, u, v, zero)
                zi = self._calc_zi(f_ord, f_ext, k)
                params = self._quarter_fit(psi, zi)
                current = {key: params[key].nominal for key in previous.keys()}
                logger.debug('quarterwave iter %i: %s', i, dict(**params, k=k))
                # check if the difference is smaller than the tolerance
                if np.allclose([current[i] for i in previous.keys()],
                               [previous[i] for i in previous.keys()],
                               atol=self.iter_tolerance):
                    converged = True
                    continue

                # update previous values
                previous = {i: current[i] for i in previous.keys()}

                # re-estimate k based on current q value
                k = self._estimate_normalize_quarter(psi, f_ord, f_ext,
                                                     current['q'])
            if not converged:
                raise RuntimeError(f'Could not converge after {self.max_iters}'
                                   ' iterations.')
        zero = params['zero']
        params['zero'] = QFloat(zero.nominal, zero.uncertainty, 'deg')
        return StokesParameters('quarterwave', **params, k=k,
                                zi=zi, psi=QFloat(psi, unit='deg'),
                                flux=fluxes)

    def compute(self, psi, f_ord, f_ext, f_ord_error=None, f_ext_error=None):
        """Compute the Stokes params from ordinary and extraordinary fluxes.

        Parameters
        ----------
        psi : array_like
            Retarder positions in degrees. Must be multiple of 22.5 degrees.
        f_ord : array_like
            Fluxes of ordinary beam. If `~astropop.math.QFloat`, the errors
            will be considered in the parameters computation.
        f_ext : array_like
            Fluxes of extraordinary beam. If `~astropop.math.QFloat`, the
            errors will be considered in the parameters computation.
        f_ord_error : array_like, optional
            Errors of the ordinary fluxes. Conflicts with the error of `f_ord`
            if it is a `~astropop.math.QFloat`.
        f_ext_error : array_like, optional
            Errors of the extraordinary fluxes. Conflicts with the error of
            `f_ext` if it is a `~astropop.math.QFloat`.

        Returns
        -------
        `~astropop.polarimetry.StokesParameters`:
            Instances containing the computed Stokes Parameters.
        """
        self._check_positions(psi)
        f_ord = QFloat(f_ord, uncertainty=f_ord_error)
        f_ext = QFloat(f_ext, uncertainty=f_ext_error)

        if self.retarder == 'halfwave':
            return self._half_compute(psi, f_ord, f_ext)
        if self.retarder == 'quarterwave':
            return self._quarter_compute(psi, f_ord, f_ext)

    @abc.abstractmethod
    def _half_fit(self, psi, zi):
        """Fit the Stokes params for halfwave retarder."""

    @abc.abstractmethod
    def _quarter_fit(self, psi, zi):
        """Fit the Stokes params for quarterwave retarder."""


@dataclass
class SLSDualBeamPolarimetry(_DualBeamPolarimetry):
    """Polarimetry computation for Stokes Least Squares algorithm.

    This method is describe in [1]_ and consists in fitting the data using
    theoretical models. This method has the advantage of not need particular
    sets of retarder positions and can handle missing points. More datailed
    description of fitted equations are given in [2]_ and [3]_.

    Parameters
    ----------
    retarder: str
        Retarder type. Must be 'quarterwave' or 'halfwave'.
    k: float (optional)
        Normalization factor. If None, it is estimated from the data.
        Default is None.
    zero: float (optional)
        Zero position of the retarder in degrees. Defult is None. If None,
        it will be estimated from the data on quarterwave retarders and will be
        zero for halfwave retarders.
    compute_k: bool (optional)
        Fit the normalization factor using the data. Default is False.
        Conflicts with ``k`` argument.
    min_snr: float (optional)
        Minimum signal-to-noise ratio. Points with lower SNR will be discarded.
        Default is None.
    psi_deviation: float (optional)
        Maximum deviation of the psi position from the sequence multiple
        of 22.5 degrees. Default is 0.1 degrees.
    iter_tolerance: float (optional)
        When fitting the parameter for quarterwave retarders, the iteration
        stops when the difference between the previous and the current
        parameters is less than this tolerance. Default is 1e-5.
    max_iters: int (optional)
        Maximum number of iterations. Default is 100.

    Notes
    -----
    - The model fitting is performed by `~scipy.optimize.cure_fit` function,
      using the Trust Region Reflective ``trf`` method.
    - If ``k`` or ``zero`` arguments are passed, they won't be computed.
      Instead, the passed values will be used.
    - If ``compute_k`` is True, the value will be estimated from the data.
      For halfwave retarders, it will be estimated by the ratio of the total
      flux of the ordinary flux and the extraordinary flux in all positions
      [2]_. For quarterwave retarders, it will be estimated in a iterative
      process described in [4]_.

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2019PASP..131b4501N
    .. [2] https://ui.adsabs.harvard.edu/abs/1984PASP...96..383M
    .. [3] https://ui.adsabs.harvard.edu/abs/1998A&A...335..979R
    .. [4] https://ui.adsabs.harvard.edu/abs/2021AJ....161..225L
    """

    def _get_fitter(self, zi):
        """Get the fitter function."""
        errors = zi.std_dev
        if np.all(errors > 0):
            fitter = partial(curve_fit, sigma=errors)
        else:
            fitter = curve_fit
            logger.info('Errors will not be considered in the fit.')
        return fitter

    def _half_fit(self, psi, zi):
        """Fit the model for halfwave retarder."""
        # change the model to do not have to fit the zero position
        model = partial(halfwave_model, zero=self.zero)
        fitter = self._get_fitter(zi)

        # fit the model
        (q, u), pcov = fitter(model, psi, zi.nominal, method='trf',
                              bounds=([-1, -1], [1, 1]))
        # Errors are the diagonal of the covariance matrix
        q_err, u_err = np.sqrt(np.diag(pcov))

        # use qfloat for fitted values
        return QFloat(q, uncertainty=q_err), QFloat(u, uncertainty=u_err)

    def _quarter_fit(self, psi, zi):
        # if zero is not set, estimate it from the data
        if self.zero is not None:
            model = partial(quarterwave_model, zero=self.zero)
            bounds = ([-1, -1, -1], [1, 1, 1])
            pnames = ['q', 'u', 'v']
            logger.debug('Using fixed value of zero: %s', self.zero)
        else:
            model = quarterwave_model
            bounds = ([-1, -1, -1, 0], [1, 1, 1, 180])
            pnames = ['q', 'u', 'v', 'zero']
            logger.debug('Zero position not set. Computing it from the data.')

        fitter = self._get_fitter(zi)
        params, pcov = fitter(model, psi, zi.nominal, method='trf',
                              bounds=bounds)
        stokes = {pnames[i]: QFloat(params[i], uncertainty=np.sqrt(pcov[i, i]))
                  for i in range(len(pnames))}
        if len(pnames) == 3:
            stokes['zero'] = QFloat(self.zero)
        return stokes
