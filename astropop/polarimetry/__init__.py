# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Compute polarimetry of dual beam polarimeters images."""

from __future__ import annotations

import typing as _t
import numpy.typing as _npt
from .._typing import NumberArray as _NumberArray, QuantityInput as _QuantityInput

import abc
import numpy as np
from dataclasses import dataclass, field
from astropy.table import Table
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from astropy import units
from functools import partial

from ..logger import logger
from ..math.physical import QFloat, convert_to_qfloat


__all__ = ['estimate_dxdy', 'match_pairs',
           'quarterwave_model', 'halfwave_model',
           'SLSDualBeamPolarimetry', 'StokesParameters']


def _compute_theta(q: _QuantityInput, u: _QuantityInput) -> QFloat | units.Quantity:
    """Compute theta using Q and U, considering quadrants and max 180 value."""
    # numpy arctan2 already looks for quadrants and is defined in [-pi, pi]
    theta = np.degrees(0.5*np.arctan2(u, q))
    if not hasattr(theta, 'unit'):
        theta = theta*units.degree
    # do not allow negative values
    if theta < 0*units.degree:
        theta += 180*units.degree
    return theta


def estimate_dxdy(
    x: _npt.ArrayLike,
    y: _npt.ArrayLike,
    steps: _t.Sequence[int]=[100, 30, 5, 3],
    bins: int=30,
    dist_limit: float=100,
) -> tuple[float, float]:
    """Estimate the displacement between the two beams.

    To compute the displacement between the ordinary and extraordinary
    beams, this function computes the most common distances between the
    sources in image, using clipped histograms around the peak.

    Parameters
    ----------
    x, y
        Arrays of x and y positions of the sources.
    steps
        Number of pixels, around the mode, to clip the histogram in each step.
    bins
        Number of bins to use in the histogram.
    dist_limit
        Maximum distance between the pairs of sources to consider.

    Return
    ------
    dx, dy
        Histogram-mode pixel differences x[i]-x[j], y[i]-y[j], using pairs
        with y[j] > y[i]. The y displacement is therefore negative. Pass
        these values to ``match_pairs`` and identify physical beams from
        the instrument convention, not the displacement sign.

    Notes
    -----
    The function constructs all detection pairs before filtering, requiring
    storage that grows quadratically with the number of detections.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.polarimetry import estimate_dxdy
    >>> rng = np.random.default_rng(42)
    >>> base_x, base_y = rng.uniform(0, 950, (2, 20))
    >>> x = np.concatenate((base_x, base_x + 25))
    >>> y = np.concatenate((base_y, base_y + 35))

    >>> # Repeated offsets identify the two-beam displacement among unrelated pairs.
    >>> dx, dy = estimate_dxdy(x, y, dist_limit=50, steps=[50, 5, 3, 1], bins=100)
    >>> print(f'{dx:.1f}, {dy:.1f} pixels')
    -25.0, -35.0 pixels
    """
    def _find_max(d: _npt.ArrayLike) -> float:
        """Refine the peak displacement using successively narrower histograms."""
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


def match_pairs(
    x: _npt.ArrayLike,
    y: _npt.ArrayLike,
    dx: float,
    dy: float,
    tolerance: float=1.0,
) -> _npt.NDArray:
    """Match the pairs of ordinary/extraordinary points (x, y).

    This function matches the pairs of ordinary and extraordinary points
    (x, y) using the displacement (dx, dy) between the two beams.

    Parameters
    ----------
    x, y
        Arrays of x and y positions of the sources.
    dx, dy
        Pixel displacement: search for a neighbor at (x-dx, y-dy) for
        every supplied detection.
    tolerance
        Maximum Euclidean distance in pixels from the predicted neighbor.

    Return
    ------
    `numpy.ndarray`:
        Structured array of indices into the input positions. Field 'o'
        contains the starting detection, and 'e' its matched neighbor.
        These geometric labels do not establish physical beam identity.

    Notes
    -----
    Independent nearest-neighbor queries do not enforce unique assignments.
    Check reused neighbors in crowded fields before extracting beam fluxes.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.polarimetry import match_pairs
    >>> x, y = np.array([10., 35.]), np.array([20., 55.])

    >>> # Search at (x - dx, y - dy), so these displacements are negative.
    >>> pairs = match_pairs(x, y, dx=-25., dy=-35., tolerance=1.)
    >>> print(pairs.tolist())
    [(0, 1)]
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


def quarterwave_model(
    psi: _QuantityInput,
    q: _QuantityInput,
    u: _QuantityInput,
    v: _QuantityInput,
    zero: _QuantityInput | None=0,
) -> _NumberArray | QFloat:
    """Compute polarimetry z(psi) model for quarter wavelength retarder.

    Z(psi) = Q*cos(2psi)**2 + U*sin(2psi)*cos(2psi) - V*sin(2psi)

    Parameters
    ----------
    psi
        Array of retarder positions in degrees.
    q, u, v
        Stokes parameters
    zero
        Zero position of the retarder in degrees.

    Return
    ------
    z: array_like
        Array of polarimetry values.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.polarimetry import quarterwave_model
    >>> psi = np.array([0., 45., 90., 135.])
    >>> modulation = quarterwave_model(psi, q=0.1, u=0.2, v=0.03)
    >>> print(np.round(modulation, 3).tolist())
    [0.1, -0.03, 0.1, 0.03]
    """
    if zero is not None:
        psi = psi+zero  # avoid inplace modification
    psi = np.radians(psi)
    psi2 = 2*psi
    zi = q*(np.cos(psi2)**2) + u*np.sin(psi2)*np.cos(psi2) - v*np.sin(psi2)
    return zi


def halfwave_model(
    psi: _QuantityInput,
    q: _QuantityInput,
    u: _QuantityInput,
    zero: _QuantityInput | None=None,
) -> _NumberArray | QFloat:
    """Compute polarimetry z(psi) model for half wavelength retarder.

    Z(psi) = Q*cos(4*psi) + U*sin(4*psi)

    Parameters
    ----------
    psi
        Array of retarder positions in degrees.
    q, u
        Stokes parameters
    zero
        Zero angle of the retarder position.

    Return
    ------
    z: array_like
        Array of polarimetry values.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.polarimetry import halfwave_model
    >>> psi = np.array([0., 22.5, 45., 67.5])
    >>> modulation = halfwave_model(psi, q=0.1, u=0.2)
    >>> print(np.round(modulation, 3).tolist())
    [0.1, 0.2, -0.1, -0.2]
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
    q, u, v
        Stokes parameters.
    retarder
        'quarterwave' or 'halfwave'
    k
        Normalization constant used to compute zi.
    zero
        Zero position of the retarder in degrees.
    zi
        Relative difference of fluxes between ordinary and extraordinary beams.
    psi
        Array of retarder positions in degrees.
    flux
        Sum of ordinary and extraordinary counts in each retarder position.

    Examples
    --------
    >>> from astropop.polarimetry import StokesParameters
    >>> result = StokesParameters('halfwave', q=0.03, u=0.04)

    >>> # Linear polarization combines q and u; theta uses the half-angle convention.
    >>> print(f'{100 * result.p.nominal:.1f}%')
    5.0%
    >>> print(f'{result.theta.nominal:.2f} {result.theta.unit}')
    26.57 deg
    """

    retarder: _t.Literal['halfwave', 'quarterwave']  # 'quarterwave' or 'halfwave'
    q: _QuantityInput
    u: _QuantityInput
    v: _QuantityInput | None = None
    k: float | QFloat | None = 1.0
    zero: _QuantityInput | None = 0.0
    flux: QFloat | None = None
    zi: _QuantityInput | None = None
    psi: _QuantityInput | None = None

    def __post_init__(self) -> None:
        """Validate the retarder and normalize Stokes values, angles and flux inputs."""
        # Check if all variables are correct
        self.retarder = str(self.retarder)
        if self.retarder not in ('halfwave', 'quarterwave'):
            raise ValueError('retarder must be halfwave or quarterwave')

        self.q = convert_to_qfloat(self.q)
        self.u = convert_to_qfloat(self.u)
        if self.v is not None:
            self.v = convert_to_qfloat(self.v)

        if self.k is not None:
            self.k = float(self.k)

        # TODO: make the angle checking smarter
        if self.zero is not None:
            self.zero = convert_to_qfloat(self.zero)
            if self.zero.unit == units.dimensionless_unscaled:
                self.zero.unit = 'deg'

        if self.zi is not None:
            self.zi = convert_to_qfloat(self.zi)
        if self.psi is not None:
            self.psi = convert_to_qfloat(self.psi)
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
    def p(self) -> QFloat:
        """Linear polarization level."""
        return np.hypot(self.q, self.u)

    @property
    def theta(self) -> QFloat:
        """Linear polarization position angle in degrees, modulo 180.

        Uses half of atan2(u, q) to preserve the quadrant. The approximate
        uncertainty is 28.6 * p.std_dev / p.nominal degrees and is undefined
        for zero linear polarization.
        """
        theta = _compute_theta(self.q, self.u).nominal
        err = 28.6*self.p.std_dev/self.p.nominal
        return QFloat(theta, err, 'deg')

    @property
    def rms(self) -> _NumberArray:
        """Population standard deviation of modulation residuals (ddof=0).

        Despite the property name, this is not the root mean square: a
        constant residual offset does not contribute. Requires zi and psi.
        """
        if self.zi is None or self.psi is None:
            raise ValueError('StokesParameters without zi and psi data has no '
                             'fitting rms')
        return np.std(self.zi-self.model(self.psi))

    @property
    def model(self) -> _t.Callable[[_QuantityInput], _NumberArray | QFloat]:
        """Callable model of the modulation (zi)."""
        if self.retarder == 'quarterwave':
            model = partial(quarterwave_model, q=self.q, u=self.u, v=self.v,
                            zero=self.zero)
        else:
            model = partial(halfwave_model, q=self.q, u=self.u, zero=self.zero)
        return model

    @property
    def theor_sigma(self) -> dict[str, _NumberArray]:
        """Theoretical error of stokes parameters from the fluxes SNR."""
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

    def __repr__(self) -> str:
        """Identify this result object and display its Stokes values."""
        return object.__repr__(self) + '\n' + str(self)

    def __str__(self) -> str:
        """Format q and u, including v only when circular polarization is present."""
        s = f'q={self.q}, u={self.u}'
        if self.v is not None:
            s += f', v={self.v}'
        return s


@dataclass
class _DualBeamPolarimetry(abc.ABC):
    """Base class for polarimetry computation."""

    retarder: _t.Literal['halfwave', 'quarterwave']  # 'quarterwave' or 'halfwave'
    k: float | None = None  # global normalization constant
    zero: float | QFloat | units.Quantity | None = None  # zero position of the retarder. If not set, computed.
    compute_k: bool = False  # compute the normalization constant
    min_snr: float | None = None  # minimum signal-to-noise ratio
    psi_deviation: float = 0.1  # max deviation of retarder position
    iter_tolerance: float = 1e-5  # maximum tolerance on the iterative fitting
    max_iters: int = 100  # maximum number of iterations
    zero_range: _t.Sequence[float] = field(default_factory=lambda: [0, 90])  # range of the zero position

    def __post_init__(self) -> None:
        """Validate retarder and normalization settings and normalize angle bounds."""
        if self.retarder not in ['quarterwave', 'halfwave']:
            raise ValueError(f"Retarder {self.retarder} unknown.")
        if isinstance(self.zero, (QFloat, units.Quantity)):
            self.zero = self.zero.to(units.degree).value
        if self.k is not None and self.compute_k:
            raise ValueError('k and compute_k cannot be used together.')
        if not self.compute_k:
            logger.info('Normalization disabled.')

        self.zero_range = list(self.zero_range)
        if not len(self.zero_range) == 2:
            raise ValueError('Zero range must be a list with two elements.')

        # number of positions per cicle
        self._n_pos = 8 if self.retarder == 'quarterwave' else 4

    def _check_positions(self, psi: _npt.ArrayLike) -> None:
        """Check if positions are in the correct order."""
        # Only positions multiple of 22.5 degrees are allowed
        devs = np.abs(psi/22.5 - np.round(psi/22.5, 0))*22.5
        if np.any(devs > self.psi_deviation):
            raise ValueError("Retarder positions must be multiple of 22.5 deg")

    def _calc_zi(self, f_ord: QFloat, f_ext: QFloat, k: float) -> QFloat:
        """Compute zi from ordinary and extraordinary fluxes."""
        return (f_ord - f_ext*k)/(f_ord + f_ext*k)

    def _estimate_normalize_half(
        self,
        psi: _npt.ArrayLike,
        f_ord: QFloat,
        f_ext: QFloat,
    ) -> float:
        """Estimate the normalization factor for halfwave retarder."""
        pos_in_cycle = np.mod(np.floor_divide(psi, 22.5), self._n_pos)
        pos_in_cycle = pos_in_cycle.astype(int)
        ford_mean = np.full(self._n_pos, np.nan)
        fext_mean = np.full(self._n_pos, np.nan)

        for i in range(self._n_pos):
            if np.all(np.isnan(f_ord[pos_in_cycle == i])) or \
               np.all(np.isnan(f_ext[pos_in_cycle == i])):
                raise ValueError('Could not estimate the normalization '
                                 'factor.')
            ford_mean[i] = np.nanmedian(f_ord[pos_in_cycle == i])
            fext_mean[i] = np.nanmedian(f_ext[pos_in_cycle == i])

        # use the means for each position. This fixes problems with missing
        # points
        return np.sum(ford_mean)/np.sum(fext_mean)

    def _estimate_normalize_quarter(
        self,
        psi: _npt.ArrayLike,
        f_ord: QFloat,
        f_ext: QFloat,
        q: float,
    ) -> float:
        """Estimate the normalization factor for quarterwave retarder."""
        ratio = self._estimate_normalize_half(psi, f_ord, f_ext)
        # PCCDPACK uses the following if to calculate the normalization. But
        # it seems that the models are not fitting correctly for combinations
        # outside this case. So, bypass it
        # q_norm = (1+0.5*q)/(1-0.5*q)
        # if (ratio < 1 and q < 0) or (ratio > 1 and q > 0):
        #     q_norm = 1/q_norm
        q_norm = (1-0.5*q)/(1+0.5*q)
        return ratio*q_norm

    def _half_compute(
        self,
        psi: _npt.ArrayLike,
        f_ord: QFloat,
        f_ext: QFloat,
    ) -> StokesParameters:
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

    def _quarter_compute(
        self,
        psi: _npt.ArrayLike,
        f_ord: QFloat,
        f_ext: QFloat,
    ) -> StokesParameters:
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

    def compute(
        self,
        psi: _npt.ArrayLike,
        f_ord: _QuantityInput,
        f_ext: _QuantityInput,
        f_ord_error: _npt.ArrayLike | None=None,
        f_ext_error: _npt.ArrayLike | None=None,
    ) -> StokesParameters:
        """Compute the Stokes params from ordinary and extraordinary fluxes.

        Parameters
        ----------
        psi
            Retarder positions in degrees. Must be multiple of 22.5 degrees.
        f_ord
            Fluxes of ordinary beam. If `~astropop.math.physical.QFloat`, the errors
            will be considered in the parameters computation.
        f_ext
            Fluxes of extraordinary beam. If `~astropop.math.physical.QFloat`, the
            errors will be considered in the parameters computation.
        f_ord_error
            Errors of the ordinary fluxes. Conflicts with the error of ``f_ord``
            if it is a `~astropop.math.physical.QFloat`.
        f_ext_error
            Errors of the extraordinary fluxes. Conflicts with the error of
            ``f_ext`` if it is a `~astropop.math.physical.QFloat`.

        Returns
        -------
        result
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
    def _half_fit(self, psi: _npt.ArrayLike, zi: QFloat) -> tuple[QFloat, QFloat]:
        """Fit the Stokes params for halfwave retarder."""

    @abc.abstractmethod
    def _quarter_fit(self, psi: _npt.ArrayLike, zi: QFloat) -> dict[str, QFloat]:
        """Fit the Stokes params for quarterwave retarder."""


@dataclass
class SLSDualBeamPolarimetry(_DualBeamPolarimetry):
    """Polarimetry computation for Stokes Least Squares algorithm.

    Fit theoretical modulation models as described in [1]_, [2]_ and [3]_.
    Retarder angles must be multiples of 22.5 degrees within psi_deviation.
    Missing positions are possible if the remaining measurements constrain
    the fit, but normalization estimation requires every phase in a cycle.

    Parameters
    ----------
    retarder
        Retarder type. Must be 'quarterwave' or 'halfwave'.
    k
        Normalization factor multiplying the extraordinary beam. Default is
        None: use 1 unless ``compute_k=True`` requests estimation.
    zero
        Zero position of the retarder in degrees. Default is None. If None,
        it will be estimated from the data on quarterwave retarders and will be
        zero for halfwave retarders.
    compute_k
        Fit the normalization factor using the data. Default is False.
        Conflicts with ``k`` argument.
    min_snr
        Reserved compatibility parameter; currently does not filter measurements.
        Select usable beam measurements before calling ``compute``. Default
        is None.
    psi_deviation
        Maximum deviation of the psi position from the sequence multiple
        of 22.5 degrees. Default is 0.1 degrees.
    iter_tolerance
        Absolute tolerance passed to ``numpy.allclose`` when comparing
        successive q, u, v during quarterwave normalization estimation.
        NumPy's default relative tolerance also applies. Default is 1e-5.
    max_iters
        Iteration budget for quarterwave normalization estimation. The
        current loop continues through this budget after recording a
        converged comparison. Raises RuntimeError if no comparison meets
        the convergence criterion. Default is 100.
    zero_range
        Minimum and maximum acceptable values for the zero position, in
        degrees. Default is [0, 90].

    Notes
    -----
    - The model fitting is performed by `~scipy.optimize.curve_fit` function,
      using the Trust Region Reflective ``trf`` method.
    - If ``k`` or ``zero`` arguments are passed, they won't be computed.
      Instead, the passed values will be used.
    - If ``compute_k`` is True, the value will be estimated from the data.
      For halfwave retarders, take the median flux per phase for each beam,
      then divide the sum of the ordinary medians by the sum of the
      extraordinary medians [2]_. Quarterwave retarders use the iterative
      process described in [4]_.
    - Propagated zi errors are relative weights only when all are positive;
      otherwise the whole fit is unweighted. ``curve_fit`` uses its default
      ``absolute_sigma=False``, scaling parameter covariance by residual
      variance. Returned errors are square roots of the covariance diagonal;
      off-diagonal covariances are not stored in the result.

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2019PASP..131b4501N
    .. [2] https://ui.adsabs.harvard.edu/abs/1984PASP...96..383M
    .. [3] https://ui.adsabs.harvard.edu/abs/1998A&A...335..979R
    .. [4] https://ui.adsabs.harvard.edu/abs/2021AJ....161..225L

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.polarimetry import SLSDualBeamPolarimetry, halfwave_model
    >>> psi = np.arange(8) * 22.5
    >>> modulation = halfwave_model(psi, q=0.03, u=-0.02)

    >>> # Split fixed total counts into beams with the desired normalized difference.
    >>> ordinary = 10000 * (1 + modulation) / 2
    >>> extraordinary = 10000 * (1 - modulation) / 2
    >>> solver = SLSDualBeamPolarimetry('halfwave', k=1., zero=0.)
    >>> result = solver.compute(psi, ordinary, extraordinary)
    >>> print(f'q={result.q.nominal:.3f}, u={result.u.nominal:.3f}')
    q=0.030, u=-0.020
    """

    def _get_fitter(
        self,
        zi: QFloat,
    ) -> _t.Callable[..., tuple[_npt.NDArray, _npt.NDArray]]:
        """Get the fitter function."""
        errors = zi.std_dev
        if np.all(errors > 0):
            fitter = partial(curve_fit, sigma=errors)
        else:
            fitter = curve_fit
            logger.info('Errors will not be considered in the fit.')
        return fitter

    def _half_fit(self, psi: _npt.ArrayLike, zi: QFloat) -> tuple[QFloat, QFloat]:
        """Fit the model for halfwave retarder."""
        # change the model to do not have to fit the zero position
        model = partial(halfwave_model, zero=self.zero)
        fitter = self._get_fitter(zi)

        # fit the model
        (q, u), pcov = fitter(model, psi, zi.nominal, method='trf',
                              bounds=([-1, -1], [1, 1]))
        # Errors are the diagonal of the covariance matrix
        q_err, u_err = np.sqrt(np.diag(pcov))
        logger.debug("Fitted Parameters:\n%s", (q, u))
        logger.debug("Covariance Matrix:\n%s", pcov)

        # use qfloat for fitted values
        return QFloat(q, uncertainty=q_err), QFloat(u, uncertainty=u_err)

    def _quarter_fit(self, psi: _npt.ArrayLike, zi: QFloat) -> dict[str, QFloat]:
        """Fit q, u and v, optionally solving for the retarder zero position.

        Stokes values are bounded between -1 and 1. A fitted zero is restricted to
        zero_range. Return fitted values with uncertainties from the covariance
        diagonal; a fixed zero is returned without a fitted uncertainty.
        """
        # if zero is not set, estimate it from the data
        if self.zero is not None:
            model = partial(quarterwave_model, zero=self.zero)
            bounds = ([-1, -1, -1], [1, 1, 1])
            pnames = ['q', 'u', 'v']
            logger.debug('Using fixed value of zero: %s', self.zero)
        else:
            model = quarterwave_model
            bounds = ([-1, -1, -1, self.zero_range[0]],
                      [1, 1, 1, self.zero_range[1]])
            pnames = ['q', 'u', 'v', 'zero']
            logger.debug('Zero position not set. Computing it from the data.')

        fitter = self._get_fitter(zi)
        params, pcov = fitter(model, psi, zi.nominal, method='trf',
                              bounds=bounds)
        logger.debug("Fitted Parameters %s: %s", pnames, params)
        logger.debug("Covariance Matrix:\n%s", pcov)
        stokes = {pnames[i]: QFloat(params[i], uncertainty=np.sqrt(pcov[i, i]))
                  for i in range(len(pnames))}
        if len(pnames) == 3:
            stokes['zero'] = QFloat(self.zero)
        return stokes
