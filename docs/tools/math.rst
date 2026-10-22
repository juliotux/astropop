.. include:: ../references.txt

Numerical helpers and PSF models
================================

Use :doc:`../data_structures/physical` for quantities with uncertainties. The
modules below operate on numerical arrays and model parameters; they do not
propagate FrameData units, flags, or configuration automatically.

Array coordinates and cutouts
-----------------------------

``iraf_indices`` returns coordinate grids in ``(x, y)`` order, with integer
values at pixel centers starting at zero. ``xy2r`` converts those grids into
flattened radii relative to a supplied center, returning flattened data in the
same order::

    >>> import numpy as np
    >>> from astropop.math.array import iraf_indices, xy2r, trim_array, all_equal
    >>> data = np.arange(25).reshape(5, 5)
    >>> x, y = iraf_indices(data)
    >>> radii, values = xy2r(x, y, data, 2, 2)
    >>> radii[12], values[12]
    (np.float64(0.0), np.int64(12))
    >>> cutout, cx, cy = trim_array(data, 2, (2, 2))
    >>> cutout.shape, (cx, cy)
    ((3, 3), (1, 1))
    >>> bool(all_equal(np.ones((2, 2))))
    True

``trim_array`` clips the requested region to image boundaries. Its
``box_size`` describes the width used to compute integer bounds; the inclusive
upper center can yield ``box_size + 1`` pixels for even sizes. The default
return is ``(cutout, local_x, local_y)``. With ``indices=(y_grid, x_grid)`` it
instead returns ``(cutout, x_grid_cutout, y_grid_cutout)``. ``origin`` is added
to the supplied center before selecting bounds; prefer the default zero-based
coordinates. ``all_equal`` expects a nonempty array.

Gaussian and Moffat profiles
----------------------------

:mod:`astropop.math.models` supplies Astropy fittable models for one-dimensional,
two-dimensional, and radial Gaussian/Moffat profiles. For two-dimensional
models, ``flux`` is integrated flux rather than a peak pixel value; ``sky``
is an additive constant background. Choose the profile's
parameters explicitly, and use Astropy's fitting tools if a fit is required::

    >>> from astropop.math.models import PSFGaussian2D, PSFMoffat2D
    >>> gaussian = PSFGaussian2D(x_0=2, y_0=2, flux=100, sigma_x=1, sigma_y=1, sky=10)
    >>> profile = gaussian(x, y)
    >>> print('profile.shape:', profile.shape)
    profile.shape: (5, 5)
    >>> print(f'Central pixel: {profile[2, 2]:.2f}')
    Central pixel: 25.92
    >>> moffat = PSFMoffat2D(width=2, power=2.5)
    >>> print(f'Moffat FWHM: {float(moffat.fwhm):.2f} pixels')
    Moffat FWHM: 2.26 pixels

Gaussian widths are standard deviations; Moffat ``width`` and ``power``
control the core and wings. Two-dimensional Gaussian ``theta`` is in degrees.
The one-dimensional Gaussian has one-dimensional integral normalization. Radial
Gaussian models use that same normalization as a function of radius. The
one-dimensional and radial Moffat variants instead use the historical
amplitude ``flux * 2 * (power - 1) / width**2``; do not assume their one-dimensional
integrals or area-weighted radial integrals equal ``flux``. A finite sampled
cutout need not sum to a model's flux parameter. Model and equation classes
expose the corresponding FWHM relations.

.. plot::
    :include-source: false
    :alt: Gaussian and Moffat circular 2D profiles with equal total flux and FWHM show different peak heights and extended Moffat wings; enclosed flux rises more slowly for the Moffat model.
    :caption: Both profiles have integrated flux 1000, FWHM 4 pixels and zero sky; Moffat power is 2.5. The dotted line marks half the FWHM, where each profile reaches half its own peak. Right: analytic continuous 2D integrals, not sums of pixel-center samples. The Moffat wings leave more light outside a finite aperture.

    import matplotlib.pyplot as plt
    import numpy as np

    from astropop.math.models import PSFGaussian2D, PSFMoffat2D

    fwhm, flux, power = 4., 1000., 2.5
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    width = fwhm / (2 * np.sqrt(2**(1 / power) - 1))
    radius = np.linspace(0, 5 * fwhm, 1500)
    gaussian = PSFGaussian2D(flux=flux, sigma_x=sigma, sigma_y=sigma)
    moffat = PSFMoffat2D(flux=flux, width=width, power=power)
    # Evaluate the numerical profiles directly. Sphinx resolves model type
    # annotations for display, which can alter Astropy's inferred input units.
    profiles = [model.evaluate(radius, np.zeros_like(radius), *model.parameters)
                for model in (gaussian, moffat)]
    fractions = [1 - np.exp(-radius**2 / (2 * sigma**2)),
                 1 - (1 + (radius / width)**2)**(1 - power)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    for name, profile, fraction in zip(('Gaussian', 'Moffat'), profiles, fractions):
        axes[0].semilogy(radius / fwhm, profile, label=name)
        axes[1].plot(radius / fwhm, fraction, label=name)
    axes[0].set(xlabel='Radius / FWHM', ylabel='Surface brightness (flux / pixel²)',
                title='Equal integrated flux; different peaks', ylim=(0.01, 100))
    axes[1].set(xlabel='Aperture radius / FWHM', ylabel='Enclosed flux / total flux',
                title='Continuous circular-aperture integral', ylim=(0, 1.03))
    for ax in axes:
        ax.axvline(0.5, color='0.5', linestyle=':', linewidth=1)
        ax.grid(alpha=0.2)
        ax.legend()

Short identifiers
-----------------

``hasher(value, size=8)`` stringifies its input, computes its UTF-8 SHA-256
hex digest, and returns the first ``size`` characters::

    >>> from astropop.math.hasher import hasher
    >>> hasher('abc')
    'ba7816bf'

Truncation permits collisions. These identifiers are useful for cache names;
they are not uniqueness guarantees or a password-storage interface.

Numerical API
-------------

See :doc:`../reference/utilities` for the API reference.

.. raw:: html

    <span id="astropop-math-array-module"></span>
    <span id="astropop-math-hasher-module"></span>
    <span id="astropop-math-models-module"></span>
    <span id="classes"></span>
    <span id="functions"></span>
    <span id="id1"></span>
    <span id="module-astropop.math.array"></span>
    <span id="module-astropop.math.hasher"></span>
    <span id="module-astropop.math.models"></span>
