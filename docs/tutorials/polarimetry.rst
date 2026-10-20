Recover known half-wave Stokes parameters
=========================================

Recover linear polarization from eight pairs of illustrative beam fluxes.
No files or online catalogs are required. The example assumes a half-wave
retarder, matched ordinary/extraordinary beams, unit beam normalization, and
angles in degrees. Read the beam and angle definitions in
:doc:`../reduction/polarimetry` before applying it to observations.

Prepare the beam measurements
------------------------------

Create eight retarder positions with known normalized Stokes values
``q=0.03`` and ``u=-0.02``. For half-wave data the normalized beam difference
is ``q*cos(4*psi) + u*sin(4*psi)``::

    >>> import numpy as np
    >>> from astropop.polarimetry import SLSDualBeamPolarimetry
    >>> psi = np.arange(8) * 22.5
    >>> q, u = 0.03, -0.02
    >>> z = q * np.cos(np.deg2rad(4 * psi)) + u * np.sin(np.deg2rad(4 * psi))
    >>> ordinary = 10000 * (1 + z) / 2
    >>> extraordinary = 10000 * (1 - z) / 2

Fit and inspect the polarization
---------------------------------

Use ``'halfwave'`` for a half-wave retarder, ``k=1`` for already normalized
beams and ``zero=0`` for a known zero angle of zero degrees. Supply a standard
uncertainty for each beam measurement::

    >>> solver = SLSDualBeamPolarimetry('halfwave', k=1., zero=0.)
    >>> result = solver.compute(psi, ordinary, extraordinary,
    ...                         np.sqrt(ordinary), np.sqrt(extraordinary))
    >>> print(f'q = {result.q.nominal:.3f}, u = {result.u.nominal:.3f}')
    q = 0.030, u = -0.020
    >>> print(f'Linear polarization: {100 * result.p.nominal:.2f}%')
    Linear polarization: 3.61%
    >>> print(f'Position angle: {result.theta.nominal:.2f} {result.theta.unit}')
    Position angle: 163.15 deg

Here the supplied errors are an illustrative Poisson model and the nominal data
contain no random noise. A tiny fitted residual does not prove the errors or
instrument calibration are realistic. Inspect recovered Stokes uncertainties,
beam fluxes, normalization and residuals versus retarder angle on actual data.
Avoid saturated beams and incomplete or degenerate angle coverage.

The fitted ``q`` and ``u`` recover the supplied modulation. The degree of
linear polarization is ``sqrt(q**2 + u**2)``; the position angle is half the
angle of the ``(q, u)`` vector, reported modulo 180 degrees. The quantities
``q``, ``u`` and ``p`` are fractions internally; multiply by 100 when
displaying percentages.

Inspect modulation and residuals
---------------------------------

The upper panel shows the normalized ordinary-minus-extraordinary beam signal,
with the supplied beam uncertainties propagated to each point. The curve is
computed from the recovered Stokes parameters. Half-wave modulation repeats
every 90 degrees of retarder rotation; this is why several distinct retarder
positions constrain the same linear polarization. When evaluating
``result.model`` yourself, supply angles with units; the fitted zero angle also
carries an angular unit::

    >>> from astropy import units as u
    >>> predicted = result.model(psi * u.deg)
    >>> residual = result.zi.nominal - predicted.nominal
    >>> print(f'Largest modulation residual: {np.max(np.abs(residual)):.3f}')
    Largest modulation residual: 0.000

.. plot:: _ext/guide_figures.py plot_polarimetry
    :include-source: False
    :alt: Half-wave beam modulation with measurement error bars and a fitted curve; residuals are zero for the noiseless example.
    :caption: Recovered modulation and residuals for the eight beam pairs above. Error bars describe the supplied measurement model, not scatter injected into the nominal fluxes.

The residual panel is flat because these nominal fluxes lie exactly on the
model. On observations, look for outliers, angle-dependent structure or unequal
noise between positions. A flat residual in this idealized example does not
validate an instrument's normalization or establish the accuracy of the fitted
uncertainties.

Choose the normalization and angle coverage
-------------------------------------------

Use either a supplied ``k`` or ``compute_k=True`` to estimate beam
normalization; these options cannot be used together. Retarder positions must
be multiples of 22.5 degrees for this solver. Supply a nonempty sequence with
sufficient independent angles: optimizer convergence alone does not establish
a well-constrained fit. Inspect angular coverage and covariance/uncertainties
as well as nominal values. Invalid inputs raise an exception rather than
representing an unpolarized source.

The advanced guide covers ``compute_k=True`` for fitting normalization,
quarter-wave circular polarization, zero offsets and convergence controls.
Do not fit both normalization and zero blindly for purely linear quarter-wave
data: those parameters can be degenerate. This tutorial does not implement a
full instrumental-polarization calibration.
