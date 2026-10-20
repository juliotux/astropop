Recover known half-wave Stokes parameters
=========================================

Recover linear polarization from eight pairs of illustrative beam fluxes.
The local beam table requires no online catalog. The example assumes a half-wave
retarder, matched ordinary/extraordinary beams, unit beam normalization, and
angles in degrees. Read the beam and angle definitions in
:doc:`../reduction/polarimetry` before applying it to observations.

.. tutorial-notebook:: polarimetry

.. tutorial-data::

Extract the shared archive and locate ``data/polarimetry.ecsv``. This is a
separate beam-measurement sequence, not the ordinary imaging stack.
The supplied nominal fluxes are noiseless and their errors follow an illustrative
Poisson model. They do not represent an instrumental polarization calibration.

.. code-block:: python

    from pathlib import Path
    data = Path('data')

.. doctest::
    :hide:

    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from tutorial_data import generate_dataset
    >>> _temporary = TemporaryDirectory()
    >>> data = Path(_temporary.name).resolve() / 'data'
    >>> generate_dataset(data)

Read the beam measurements
---------------------------

Inspect paired ordinary/extraordinary fluxes and their retarder angles before
fitting. These measurements have already been extracted with aperture photometry;
beam identification and instrumental corrections must precede this step on
observations::

    >>> import numpy as np
    >>> from astropy.table import Table
    >>> from astropop.polarimetry import SLSDualBeamPolarimetry
    >>> beams = Table.read(data / 'polarimetry.ecsv')
    >>> psi = np.array(beams['angle_deg'])
    >>> ordinary, extraordinary = beams['ordinary'], beams['extraordinary']
    >>> print(f'Beam pairs: {len(beams)}; angle spacing: {psi[1] - psi[0]:.1f} deg')
    Beam pairs: 8; angle spacing: 22.5 deg

Fit and inspect the polarization
---------------------------------

Use ``'halfwave'`` for a half-wave retarder, ``k=1`` for already normalized
beams and ``zero=0`` for a known zero angle of zero degrees. Supply a standard
uncertainty for each beam measurement::

    >>> solver = SLSDualBeamPolarimetry('halfwave', k=1., zero=0.)
    >>> result = solver.compute(psi, ordinary, extraordinary,
    ...                         beams['ordinary_error'], beams['extraordinary_error'])
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

The fitted ``q`` and ``u`` recover the injected fractions of 0.03 and -0.02. The degree of
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

.. plot:: _ext/workflow_figures.py plot_polarimetry
    :include-source: False
    :alt: Half-wave beam modulation with measurement error bars and a fitted curve; residuals are zero for the noiseless example.
    :caption: Recovered modulation and residuals for the eight beam pairs above. Error bars describe the supplied measurement model, not scatter injected into the nominal fluxes.

The residual panel is flat because these nominal fluxes lie exactly on the
model. On observations, look for outliers, angle-dependent structure or unequal
noise between positions. A flat residual in this idealized example does not
validate an instrument's normalization or establish the accuracy of the fitted
uncertainties.

Save the Stokes measurement
-----------------------------

Keep the polarization convention and angle units with the result::

    >>> from pathlib import Path
    >>> from astropy.table import Table
    >>> output = data.parent / 'reduced'
    >>> output.mkdir(exist_ok=True)
    >>> stokes = Table({'q': [result.q.nominal], 'u': [result.u.nominal],
    ...                 'p': [result.p.nominal], 'theta_deg': [result.theta.nominal],
    ...                 'q_error': [result.q.uncertainty], 'u_error': [result.u.uncertainty],
    ...                 'p_error': [result.p.uncertainty], 'theta_error_deg': [result.theta.uncertainty]})
    >>> stokes['theta_deg'].unit = stokes['theta_error_deg'].unit = u.deg
    >>> stokes.meta.update({'retarder': 'halfwave', 'beam_normalization': 1.,
    ...                     'zero_angle_deg': 0., 'nominal_fluxes': 'noiseless example'})
    >>> stokes.write(output / 'stokes.ecsv', overwrite=True)
    >>> print(f'Saved Stokes measurements: {len(stokes)}')
    Saved Stokes measurements: 1

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

.. doctest::
    :hide:

    >>> _temporary.cleanup()
