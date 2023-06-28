# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Math models designed for PSF."""

import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.modeling import Fittable1DModel, Fittable2DModel, Parameter


__all__ = ['MoffatEquations', 'PSFMoffat2D', 'PSFMoffat1D', 'PSFMoffatRadial',
           'GaussianEquations', 'PSFGaussian2D', 'PSFGaussian1D',
           'PSFGaussianRadial']


TWO_PI = 2*np.pi
SQRT_TWO_PI = np.sqrt(TWO_PI)


class MoffatEquations:
    """Moffat equations and utilities."""
    @staticmethod
    def normalize(width, power):
        """Normalization factor for 2D distribution."""
        return (power - 1)/(np.pi*width**2)

    @staticmethod
    def fwhm(width, power):
        """Get FWHM based on Moffat width."""
        return 2*width*np.sqrt(2**(1/power) - 1)

    @staticmethod
    def model_radial(r, width, power, flux, sky):
        """Radial Moffat model.

        Parameters:
        r: float or array
            Radial distance from the center.
        width: float
            Moffat width parameter.
        power: float
            Moffat power parameter.
        flux: float
            Total integrated flux parameter.
        sky: float
            Sky level parameter.

        Returns:
        float or array
            Values of the model at r.
        """
        # multiply by pi to normalize in 1D
        a = flux*2*(power - 1)/(width**2)
        return sky + a*(1 + (r/width)**2)**(-power)

    @staticmethod
    def model_1d(x, x0, width, power, flux, sky):
        """1D Moffat model.

        Parameters:
        x: float or array
            X coordinates to evaluate.
        x0: float
            Center coordinates of the model.
        width: float
            Moffat width parameter.
        power: float
            Moffat power parameter.
        flux: float
            Total integrated flux parameter.
        sky: float
            Sky level parameter.

        Returns:
        float or array
            Values of the model at x.
        """
        a = flux*2*(power - 1)/(width**2)
        return sky + a*(1 + ((x - x0)/width)**2)**(-power)

    @staticmethod
    def model_2d(x, y, x0, y0, width, power, flux, sky):
        """2D Moffat model.

        Parameters:
        x, y: float or array
            X and Y coordinates to evaluate.
        x0, y0: float
            Center coordinates of the model.
        width: float
            Moffat width parameter.
        power: float
            Moffat power parameter.
        flux: float
            Total integrated flux parameter.
        sky: float
            Sky level parameter.

        Returns:
        float or array
            Values of the model at (x, y).
        """
        a = flux*(power - 1)/(np.pi*width**2)
        r_sq = (x - x0)**2 + (y - y0)**2
        return sky + a*(1 + r_sq/(width**2))**(-power)


class PSFMoffat2D(Fittable2DModel):
    """2D Moffat PSF model, normalized."""

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
        return MoffatEquations.model_2d(x, y, x_0, y_0, width=width,
                                        power=power, flux=flux, sky=sky)

    @property
    def fwhm(self):
        return MoffatEquations.fwhm(width=self.width,
                                    power=self.power)


class PSFMoffat1D(Fittable1DModel):
    """1D Moffat PSF model, normalized."""

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
        return MoffatEquations.model_1d(x, x0=x_0, width=width, power=power,
                                        flux=flux, sky=sky)

    @property
    def fwhm(self):
        return MoffatEquations.fwhm(width=self.width,
                                    power=self.power)


class PSFMoffatRadial(Fittable1DModel):
    """Radial Moffat PSF model, normalized."""

    flux = Parameter(default=1, fixed=False, bounds=(0, None),
                     description='Total integrated flux')
    width = Parameter(default=1, fixed=False, bounds=(0, None),
                      description='Moffat width parameter')
    power = Parameter(default=1.5, fixed=False, bounds=(1, None),
                      description='Moffat power parameter')
    sky = Parameter(default=0, fixed=False, description='Sky level')

    @staticmethod
    def evaluate(x, flux, width, power, sky):
        return MoffatEquations.model_radial(x, width=width, power=power,
                                            flux=flux, sky=sky)

    @property
    def fwhm(self):
        return MoffatEquations.fwhm(width=self.width,
                                    power=self.power)


class GaussianEquations:
    """Gaussian equantions and utilities."""

    @staticmethod
    def normalize_2d(sigmax, sigmay):
        """Normalization factor for 2D distribution."""
        return 1/(TWO_PI*sigmax*sigmay)

    @staticmethod
    def normalize_1d(sigma):
        """Normalization factor for 1D distribution."""
        return 1/(SQRT_TWO_PI*sigma)

    @staticmethod
    def fwhm(sigma):
        """Get FWHM based on Gaussian width."""
        return sigma/gaussian_fwhm_to_sigma

    @staticmethod
    def model_radial(r, flux, sigma, sky):
        """Radial Gaussian model.

        Parameters:
        r: float or array
            Radial distance from the center.
        flux: float
            Total integrated flux parameter.
        sigma: float
            Gaussian standard deviation parameter.
        sky: float
            Sky level parameter.

        Returns:
        float or array
            Values of the model at r.
        """
        return sky + flux*np.exp(-0.5*(r**2)/(sigma**2))/(SQRT_TWO_PI*sigma)

    @staticmethod
    def model_1d(x, x0, flux, sigma, sky):
        """1D Gaussian model.

        Parameters:
        x: float or array
            X coordinates to evaluate.
        x0: float
            Center coordinates of the model.
        flux: float
            Total integrated flux parameter.
        sigma: float
            Gaussian standard deviation parameter.
        sky: float
            Sky level parameter.

        Returns:
        float or array
            Values of the model at x.
        """
        a = flux/(SQRT_TWO_PI*sigma)
        return sky + a*np.exp(-0.5*((x-x0)**2)/(sigma**2))

    @staticmethod
    def model_2d(x, y, x0, y0, flux, sigma_x, sigma_y, theta, sky):
        """2D Gaussian model.

        Parameters:
        x, y: float or array
            X and Y coordinates to evaluate.
        x0, y0: float
            Center coordinates of the model.
        sigma_x, sigma_y: float
            Gaussian standard deviation parameters.
        flux: float
            Total integrated flux parameter.
        sky: float
            Sky level parameter.

        Returns:
        float or array
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
    """2D Gaussian model, normalized."""
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
        return GaussianEquations.model_2d(x, y, x0=x_0, y0=y_0,
                                          sigma_x=sigma_x, sigma_y=sigma_y,
                                          theta=theta, flux=flux, sky=sky)

    @property
    def fwhm(self):
        """Mean FWHM of the Gaussian in X and Y."""
        return np.mean([GaussianEquations.fwhm(sigma=self.sigma_x),
                        GaussianEquations.fwhm(sigma=self.sigma_y)])


class PSFGaussian1D(Fittable1DModel):
    """1D Gaussian model, normalized."""

    x_0 = Parameter(default=0, description='X center')
    flux = Parameter(default=1, fixed=False, bounds=(0, None),
                     description='Total integrated flux')
    sigma = Parameter(default=1, fixed=False, bounds=(0, None),
                      description='Gaussian standard deviation')
    sky = Parameter(default=0, fixed=False,
                    description='Sky level')

    @staticmethod
    def evaluate(x, x_0, flux, sigma, sky):
        return GaussianEquations.model_1d(x, x0=x_0, sigma=sigma,
                                          flux=flux, sky=sky)

    @property
    def fwhm(self):
        return GaussianEquations.fwhm(sigma=self.sigma)


class PSFGaussianRadial(Fittable1DModel):
    """Radial Gaussian model, normalized."""

    flux = Parameter(default=1, fixed=False, bounds=(0, None),
                     description='Total integrated flux')
    sigma = Parameter(default=1, fixed=False, bounds=(0, None),
                      description='Gaussian standard deviation')
    sky = Parameter(default=0, fixed=False,
                    description='Sky level')

    @staticmethod
    def evaluate(x, flux, sigma, sky):
        return GaussianEquations.model_radial(x, sigma=sigma, flux=flux,
                                              sky=sky)

    @property
    def fwhm(self):
        return GaussianEquations.fwhm(sigma=self.sigma)
