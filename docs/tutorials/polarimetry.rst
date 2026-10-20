Recover known half-wave Stokes parameters
=========================================

This deterministic example executes locally under pytest. No files, catalogs,
random draws or instrument calibration are required. It assumes a half-wave
retarder, matched ordinary/extraordinary beams, unit beam normalization, and
angles in degrees. Read the beam and angle definitions in
:doc:`../reduction/polarimetry` before applying it to observations.

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
    >>> solver = SLSDualBeamPolarimetry('halfwave', k=1., zero=0.)
    >>> result = solver.compute(psi, ordinary, extraordinary,
    ...                         np.sqrt(ordinary), np.sqrt(extraordinary))
    >>> np.testing.assert_allclose(result.q.nominal, q, atol=1e-8)
    >>> np.testing.assert_allclose(result.u.nominal, u, atol=1e-8)
    >>> np.testing.assert_allclose(result.p.nominal, np.hypot(q, u), atol=1e-8)

Here the supplied errors are an illustrative Poisson model and the nominal data
contain no random noise. A tiny fitted residual does not prove the errors or
instrument calibration are realistic. Inspect recovered Stokes uncertainties,
beam fluxes, normalization and residuals versus retarder angle on actual data.
Avoid saturated beams and incomplete or degenerate angle coverage.

Invalid settings should fail explicitly rather than being interpreted as an
unpolarized source::

    >>> import pytest
    >>> with pytest.raises(ValueError, match='unknown'):
    ...     SLSDualBeamPolarimetry('invalid')
    >>> with pytest.raises(ValueError, match='cannot be used together'):
    ...     SLSDualBeamPolarimetry('halfwave', k=1., compute_k=True)
    >>> with pytest.raises(ValueError, match='multiple of 22.5'):
    ...     solver.compute(np.array([1.]), np.array([100.]), np.array([100.]))
    >>> with pytest.raises(TypeError):
    ...     solver.compute(np.array([]), np.array([]), np.array([]))

The empty sequence currently fails during input conversion rather than returning
a Stokes estimate. A nonempty sequence can still have insufficient independent
angles: optimizer convergence alone does not establish a well-constrained fit.
Inspect angular coverage and covariance/uncertainties as well as nominal values.

The advanced guide covers ``compute_k=True`` for fitting normalization,
quarter-wave circular polarization, zero offsets and convergence controls.
Do not fit both normalization and zero blindly for purely linear quarter-wave
data: those parameters can be degenerate. This tutorial does not implement a
full instrumental-polarization calibration.
