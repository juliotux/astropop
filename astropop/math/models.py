# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Math models designed for PSF."""

from __future__ import annotations

import numpy.typing as _npt
from .._typing import NumberArray as _NumberArray

import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.modeling import Fittable1DModel, Fittable2DModel, Parameter


__all__ = ['MoffatEquations', 'PSFMoffat2D', 'PSFMoffat1D', 'PSFMoffatRadial',
           'GaussianEquations', 'PSFGaussian2D', 'PSFGaussian1D',
           'PSFGaussianRadial']


TWO_PI = 2*np.pi
SQRT_TWO_PI = np.sqrt(TWO_PI)


class MoffatEquations:
    """Moffat equations and utilities.

    Examples
    --------
    >>> from astropop.math.models import MoffatEquations
    >>> print(f'{MoffatEquations.fwhm(width=2., power=2.5):.3f}')
    2.261
    """
    @staticmethod
    def normalize(width: _npt.ArrayLike, power: _npt.ArrayLike) -> _NumberArray:
        """Normalization factor for 2D distribution."""
        return (power - 1)/(np.pi*width**2)

    @staticmethod
    def fwhm(width: _npt.ArrayLike, power: _npt.ArrayLike) -> _NumberArray:
        """Get FWHM based on Moffat width."""
        return 2*width*np.sqrt(2**(1/power) - 1)

    @staticmethod
    def model_radial(
        r: _npt.ArrayLike,
        width: _npt.ArrayLike,
        power: _npt.ArrayLike,
        flux: _npt.ArrayLike,
        sky: _npt.ArrayLike,
    ) -> _NumberArray:
        """Radial Moffat model.

        Parameters
        ----------
        r
            Radial distance from the center.
        width
            Moffat width parameter.
        power
            Moffat power parameter.
        flux
            Amplitude parameter; the peak above sky is
            ``flux * 2 * (power - 1) / width**2``, not a 1D integral.
        sky
            Sky level parameter.

        Returns
        -------
        values
            Values of the model at r.
        """
        # multiply by pi to normalize in 1D
        a = flux*2*(power - 1)/(width**2)
        return sky + a*(1 + (r/width)**2)**(-power)

    @staticmethod
    def model_1d(
        x: _npt.ArrayLike,
        x0: _npt.ArrayLike,
        width: _npt.ArrayLike,
        power: _npt.ArrayLike,
        flux: _npt.ArrayLike,
        sky: _npt.ArrayLike,
    ) -> _NumberArray:
        """1D Moffat model.

        Parameters
        ----------
        x
            X coordinates to evaluate.
        x0
            Center coordinates of the model.
        width
            Moffat width parameter.
        power
            Moffat power parameter.
        flux
            Amplitude parameter; the peak above sky is
            ``flux * 2 * (power - 1) / width**2``, not a 1D integral.
        sky
            Sky level parameter.

        Returns
        -------
        values
            Values of the model at x.
        """
        a = flux*2*(power - 1)/(width**2)
        return sky + a*(1 + ((x - x0)/width)**2)**(-power)

    @staticmethod
    def model_2d(
        x: _npt.ArrayLike,
        y: _npt.ArrayLike,
        x0: _npt.ArrayLike,
        y0: _npt.ArrayLike,
        width: _npt.ArrayLike,
        power: _npt.ArrayLike,
        flux: _npt.ArrayLike,
        sky: _npt.ArrayLike,
    ) -> _NumberArray:
        """2D Moffat model.

        Parameters
        ----------
        x, y
            X and Y coordinates to evaluate.
        x0, y0
            Center coordinates of the model.
        width
            Moffat width parameter.
        power
            Moffat power parameter.
        flux
            Total integrated flux parameter.
        sky
            Sky level parameter.

        Returns
        -------
        values
            Values of the model at (x, y).
        """
        a = flux*(power - 1)/(np.pi*width**2)
        r_sq = (x - x0)**2 + (y - y0)**2
        return sky + a*(1 + r_sq/(width**2))**(-power)


# Astropy interprets evaluate annotations as units. Use PEP 484 type comments
# on those six methods to keep static types without changing unit handling.
class PSFMoffat2D(Fittable2DModel):
    """2D Moffat profile with an additive constant sky.

    Parameters
    ----------
    x_0, y_0 : float, optional
        Profile center in coordinate units. Default is 0 for each
        axis.
    flux : float, optional
        Integrated profile flux above sky on the infinite plane.
        Default is 1.
    width : float, optional
        Positive core width in coordinate units. Default is 1.
    power : float, optional
        Wing power, greater than 1 for the normalization. Default is
        1.5.
    sky : float, optional
        Additive background in output-value units. Default is 0.

    Notes
    -----
    Calling the model evaluates numerical coordinates using Astropy model
    broadcasting and returns numerical values of the corresponding shape.
    It does not integrate over detector pixels or propagate FrameData quality
    arrays or uncertainties. Use consistent coordinate and output scales.
    Declared fit bounds do not replace checking physically valid parameters.
    A finite sampled cutout need not sum to the flux parameter.

    Examples
    --------
    >>> from astropop.math.models import PSFMoffat2D
    >>> profile = PSFMoffat2D(flux=100., sky=10., width=2., power=2.5)

    >>> # The 2D flux is an integral over the plane, not the central pixel value.
    >>> print(f'{profile(0., 0.):.2f}')
    21.94
    """

    x_0 = Parameter(default=0, description='X center')
    y_0 = Parameter(default=0, description='Y center')
    flux = Parameter(default=1, fixed=False,
                     description='Total integrated flux',
                     bounds=(0, None))
    width = Parameter(default=1, fixed=False, bounds=(0, None),
                      description='Moffat width parameter',)
    power = Parameter(default=1.5, fixed=False, bounds=(1, None),
                      description='Moffat power parameter')
    sky = Parameter(default=0, fixed=False,
                    description='Sky level')

    @staticmethod
    def evaluate(x, y, x_0, y_0, flux, width, power, sky):
        # type: (_npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike) -> _NumberArray
        """Evaluate the normalized profile and additive sky at the supplied coordinates."""
        return MoffatEquations.model_2d(x, y, x_0, y_0, width=width,
                                        power=power, flux=flux, sky=sky)

    @property
    def fwhm(self) -> _NumberArray:
        """Full width at half maximum in the model coordinate units."""
        return MoffatEquations.fwhm(width=self.width,
                                    power=self.power)


class PSFMoffat1D(Fittable1DModel):
    """1D Moffat profile with an additive constant sky.

    Parameters
    ----------
    x_0 : float, optional
        Profile center in coordinate units. Default is 0 for each
        axis.
    flux : float, optional
        Amplitude coefficient, not the integral over a line or plane.
        Default is 1.
    width : float, optional
        Positive core width in coordinate units. Default is 1.
    power : float, optional
        Wing power, greater than 1 for the normalization. Default is
        1.5.
    sky : float, optional
        Additive background in output-value units. Default is 0.

    Notes
    -----
    Calling the model evaluates numerical coordinates using Astropy model
    broadcasting and returns numerical values of the corresponding shape.
    It does not integrate over detector pixels or propagate FrameData quality
    arrays or uncertainties. Use consistent coordinate and output scales.
    Declared fit bounds do not replace checking physically valid parameters.
    A finite sampled cutout need not sum to the flux parameter.
    The historical amplitude is ``flux * 2 * (power - 1) / width**2``;
    despite the parameter name, it is not a 1D integral normalization.

    Examples
    --------
    >>> from astropop.math.models import PSFMoffat1D
    >>> profile = PSFMoffat1D(flux=100., sky=10., width=2., power=2.5)

    >>> # Here flux is an amplitude coefficient rather than a one-dimensional integral.
    >>> print(f'{profile(0.):.2f}')
    85.00
    """

    x_0 = Parameter(default=0, description='X center')
    flux = Parameter(default=1, fixed=False, bounds=(0, None),
                     description='Total integrated flux')
    width = Parameter(default=1, fixed=False, bounds=(0, None),
                      description='Moffat width parameter',)
    power = Parameter(default=1.5, fixed=False, bounds=(1, None),
                      description='Moffat power parameter')
    sky = Parameter(default=0, fixed=False, description='Sky level')

    @staticmethod
    def evaluate(x, x_0, flux, width, power, sky):
        # type: (_npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike) -> _NumberArray
        """Evaluate the normalized profile and additive sky at the supplied coordinates."""
        return MoffatEquations.model_1d(x, x0=x_0, width=width, power=power,
                                        flux=flux, sky=sky)

    @property
    def fwhm(self) -> _NumberArray:
        """Full width at half maximum in the model coordinate units."""
        return MoffatEquations.fwhm(width=self.width,
                                    power=self.power)


class PSFMoffatRadial(Fittable1DModel):
    """Radial Moffat profile with an additive constant sky.

    Parameters
    ----------
    flux : float, optional
        Amplitude coefficient, not the integral over a line or plane.
        Default is 1.
    width : float, optional
        Positive core width in coordinate units. Default is 1.
    power : float, optional
        Wing power, greater than 1 for the normalization. Default is
        1.5.
    sky : float, optional
        Additive background in output-value units. Default is 0.

    Notes
    -----
    Calling the model evaluates numerical coordinates using Astropy model
    broadcasting and returns numerical values of the corresponding shape.
    It does not integrate over detector pixels or propagate FrameData quality
    arrays or uncertainties. Use consistent coordinate and output scales.
    Declared fit bounds do not replace checking physically valid parameters.
    A finite sampled cutout need not sum to the flux parameter.
    The historical amplitude is ``flux * 2 * (power - 1) / width**2``;
    despite the parameter name, it is not a 1D integral normalization.

    Examples
    --------
    >>> from astropop.math.models import PSFMoffatRadial
    >>> profile = PSFMoffatRadial(flux=100., sky=10., width=2., power=2.5)

    >>> # This radial helper uses the Moffat amplitude coefficient, not a flux integral.
    >>> print(f'{profile(0.):.2f}')
    85.00
    """

    flux = Parameter(default=1, fixed=False, bounds=(0, None),
                     description='Total integrated flux')
    width = Parameter(default=1, fixed=False, bounds=(0, None),
                      description='Moffat width parameter')
    power = Parameter(default=1.5, fixed=False, bounds=(1, None),
                      description='Moffat power parameter')
    sky = Parameter(default=0, fixed=False, description='Sky level')

    @staticmethod
    def evaluate(x, flux, width, power, sky):
        # type: (_npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike) -> _NumberArray
        """Evaluate the normalized profile and additive sky at the supplied coordinates."""
        return MoffatEquations.model_radial(x, width=width, power=power,
                                            flux=flux, sky=sky)

    @property
    def fwhm(self) -> _NumberArray:
        """Full width at half maximum in the model coordinate units."""
        return MoffatEquations.fwhm(width=self.width,
                                    power=self.power)


class GaussianEquations:
    """Gaussian equantions and utilities.

    Examples
    --------
    >>> from astropop.math.models import GaussianEquations

    >>> # Convert Gaussian sigma to full width at half maximum in the same unit.
    >>> print(f'{GaussianEquations.fwhm(2.):.3f}')
    4.710
    """

    @staticmethod
    def normalize_2d(sigmax: _npt.ArrayLike, sigmay: _npt.ArrayLike) -> _NumberArray:
        """Normalization factor for 2D distribution."""
        return 1/(TWO_PI*sigmax*sigmay)

    @staticmethod
    def normalize_1d(sigma: _npt.ArrayLike) -> _NumberArray:
        """Normalization factor for 1D distribution."""
        return 1/(SQRT_TWO_PI*sigma)

    @staticmethod
    def fwhm(sigma: _npt.ArrayLike) -> _NumberArray:
        """Get FWHM based on Gaussian width."""
        return sigma/gaussian_fwhm_to_sigma

    @staticmethod
    def model_radial(
        r: _npt.ArrayLike,
        flux: _npt.ArrayLike,
        sigma: _npt.ArrayLike,
        sky: _npt.ArrayLike,
    ) -> _NumberArray:
        """Radial Gaussian model.

        Parameters
        ----------
        r
            Radial distance from the center.
        flux
            Amplitude with 1D Gaussian normalization; not the area-weighted
            integral of a circular 2D profile.
        sigma
            Gaussian standard deviation parameter.
        sky
            Sky level parameter.

        Returns
        -------
        values
            Values of the model at r.
        """
        return sky + flux*np.exp(-0.5*(r**2)/(sigma**2))/(SQRT_TWO_PI*sigma)

    @staticmethod
    def model_1d(
        x: _npt.ArrayLike,
        x0: _npt.ArrayLike,
        flux: _npt.ArrayLike,
        sigma: _npt.ArrayLike,
        sky: _npt.ArrayLike,
    ) -> _NumberArray:
        """1D Gaussian model.

        Parameters
        ----------
        x
            X coordinates to evaluate.
        x0
            Center coordinates of the model.
        flux
            Total integrated flux parameter.
        sigma
            Gaussian standard deviation parameter.
        sky
            Sky level parameter.

        Returns
        -------
        values
            Values of the model at x.
        """
        a = flux/(SQRT_TWO_PI*sigma)
        return sky + a*np.exp(-0.5*((x-x0)**2)/(sigma**2))

    @staticmethod
    def model_2d(
        x: _npt.ArrayLike,
        y: _npt.ArrayLike,
        x0: _npt.ArrayLike,
        y0: _npt.ArrayLike,
        flux: _npt.ArrayLike,
        sigma_x: _npt.ArrayLike,
        sigma_y: _npt.ArrayLike,
        theta: _npt.ArrayLike,
        sky: _npt.ArrayLike,
    ) -> _NumberArray:
        """2D Gaussian model.

        Parameters
        ----------
        x, y
            X and Y coordinates to evaluate.
        x0, y0
            Center coordinates of the model.
        sigma_x, sigma_y
            Gaussian standard deviation parameters.
        flux
            Total integrated flux parameter.
        sky
            Sky level parameter.

        Returns
        -------
        values
            Values of the model at (x, y).
        """
        theta = np.radians(theta)
        xstd2 = sigma_x**2
        ystd2 = sigma_y**2
        cost2 = np.cos(theta)**2
        sint2 = np.sin(theta)**2
        sin2t = np.sin(2*theta)
        a = (cost2/xstd2) + (sint2/ystd2)
        b = (sin2t/xstd2) - (sin2t/ystd2)
        c = (sint2/xstd2) + (cost2/ystd2)
        xi = x - x0
        yi = y - y0
        amp = flux/(TWO_PI*sigma_x*sigma_y)
        return amp*np.exp(-0.5*((a*xi**2) + (b*xi*yi) + (c*yi**2))) + sky


class PSFGaussian2D(Fittable2DModel):
    """2D Gaussian profile with an additive constant sky.

    Parameters
    ----------
    x_0, y_0 : float, optional
        Profile center in coordinate units. Default is 0 for each
        axis.
    flux : float, optional
        Integrated profile flux above sky on the infinite plane.
        Default is 1.
    sigma_x, sigma_y : float, optional
        Positive standard deviation in coordinate units. Default is 1.
    theta : float, optional
        Profile rotation in degrees. Default is 0.
    sky : float, optional
        Additive background in output-value units. Default is 0.

    Notes
    -----
    Calling the model evaluates numerical coordinates using Astropy model
    broadcasting and returns numerical values of the corresponding shape.
    It does not integrate over detector pixels or propagate FrameData quality
    arrays or uncertainties. Use consistent coordinate and output scales.
    Declared fit bounds do not replace checking physically valid parameters.
    A finite sampled cutout need not sum to the flux parameter.

    Examples
    --------
    >>> from astropop.math.models import PSFGaussian2D
    >>> profile = PSFGaussian2D(flux=100., sky=10., sigma_x=1., sigma_y=1.)

    >>> # The 2D flux is an integral over the plane, not the central pixel value.
    >>> print(f'{profile(0., 0.):.2f}')
    25.92
    """
    x_0 = Parameter(default=0, description='X center')
    y_0 = Parameter(default=0, description='Y center')
    flux = Parameter(default=1, fixed=False, bounds=(0, None),
                     description='Total integrated flux')
    sigma_x = Parameter(default=1, fixed=False, bounds=(0, None),
                        description='Gaussian standard deviation in X')
    sigma_y = Parameter(default=1, fixed=False, bounds=(0, None),
                        description='Gaussian standard deviation in Y')
    theta = Parameter(default=0, fixed=False, bounds=(0, 360),
                      description='Position angle')
    sky = Parameter(default=0, fixed=False,
                    description='Sky level')

    @staticmethod
    def evaluate(x, y, x_0, y_0, flux, sigma_x, sigma_y, theta, sky):
        # type: (_npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike) -> _NumberArray
        """Evaluate the normalized profile and additive sky at the supplied coordinates."""
        return GaussianEquations.model_2d(x, y, x0=x_0, y0=y_0,
                                          sigma_x=sigma_x, sigma_y=sigma_y,
                                          theta=theta, flux=flux, sky=sky)

    @property
    def fwhm(self) -> _NumberArray:
        """Mean FWHM of the Gaussian in X and Y."""
        return np.mean([GaussianEquations.fwhm(sigma=self.sigma_x),
                        GaussianEquations.fwhm(sigma=self.sigma_y)])


class PSFGaussian1D(Fittable1DModel):
    """1D Gaussian profile with an additive constant sky.

    Parameters
    ----------
    x_0 : float, optional
        Profile center in coordinate units. Default is 0 for each
        axis.
    flux : float, optional
        Integral above sky on the infinite line. Default is 1.
    sigma : float, optional
        Positive standard deviation in coordinate units. Default is 1.
    sky : float, optional
        Additive background in output-value units. Default is 0.

    Notes
    -----
    Calling the model evaluates numerical coordinates using Astropy model
    broadcasting and returns numerical values of the corresponding shape.
    It does not integrate over detector pixels or propagate FrameData quality
    arrays or uncertainties. Use consistent coordinate and output scales.
    Declared fit bounds do not replace checking physically valid parameters.
    A finite sampled cutout need not sum to the flux parameter.

    Examples
    --------
    >>> from astropop.math.models import PSFGaussian1D
    >>> profile = PSFGaussian1D(flux=100., sky=10., sigma=1.)

    >>> # The 1D flux is an integral along the line; sky adds a constant offset.
    >>> print(f'{profile(0.):.2f}')
    49.89
    """

    x_0 = Parameter(default=0, description='X center')
    flux = Parameter(default=1, fixed=False, bounds=(0, None),
                     description='Total integrated flux')
    sigma = Parameter(default=1, fixed=False, bounds=(0, None),
                      description='Gaussian standard deviation')
    sky = Parameter(default=0, fixed=False,
                    description='Sky level')

    @staticmethod
    def evaluate(x, x_0, flux, sigma, sky):
        # type: (_npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike) -> _NumberArray
        """Evaluate the normalized profile and additive sky at the supplied coordinates."""
        return GaussianEquations.model_1d(x, x0=x_0, sigma=sigma,
                                          flux=flux, sky=sky)

    @property
    def fwhm(self) -> _NumberArray:
        """Full width at half maximum in the model coordinate units."""
        return GaussianEquations.fwhm(sigma=self.sigma)


class PSFGaussianRadial(Fittable1DModel):
    """Radial Gaussian profile with an additive constant sky.

    Parameters
    ----------
    flux : float, optional
        Uses the 1D normalization, not an area-weighted radial
        integral. Default is 1.
    sigma : float, optional
        Positive standard deviation in coordinate units. Default is 1.
    sky : float, optional
        Additive background in output-value units. Default is 0.

    Notes
    -----
    Calling the model evaluates numerical coordinates using Astropy model
    broadcasting and returns numerical values of the corresponding shape.
    It does not integrate over detector pixels or propagate FrameData quality
    arrays or uncertainties. Use consistent coordinate and output scales.
    Declared fit bounds do not replace checking physically valid parameters.
    A finite sampled cutout need not sum to the flux parameter.

    Examples
    --------
    >>> from astropop.math.models import PSFGaussianRadial
    >>> profile = PSFGaussianRadial(flux=100., sky=10., sigma=1.)

    >>> # This radial helper uses one-dimensional Gaussian normalization.
    >>> print(f'{profile(0.):.2f}')
    49.89
    """

    flux = Parameter(default=1, fixed=False, bounds=(0, None),
                     description='Total integrated flux')
    sigma = Parameter(default=1, fixed=False, bounds=(0, None),
                      description='Gaussian standard deviation')
    sky = Parameter(default=0, fixed=False,
                    description='Sky level')

    @staticmethod
    def evaluate(x, flux, sigma, sky):
        # type: (_npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike, _npt.ArrayLike) -> _NumberArray
        """Evaluate the normalized profile and additive sky at the supplied coordinates."""
        return GaussianEquations.model_radial(x, sigma=sigma, flux=flux,
                                              sky=sky)

    @property
    def fwhm(self) -> _NumberArray:
        """Full width at half maximum in the model coordinate units."""
        return GaussianEquations.fwhm(sigma=self.sigma)
