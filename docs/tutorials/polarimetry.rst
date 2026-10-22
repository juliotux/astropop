Recover Stokes parameters from paired beam fluxes
=================================================

Recover linear polarization from eight illustrative half-wave beam pairs, then
fit linear and circular polarization in a separate noisy quarter-wave sequence.
The local beam table requires no online catalog. The first example uses a half-wave
retarder and unit beam normalization; the second uses a quarter-wave retarder
with known calibration constants. Both require matched ordinary/extraordinary
beams and angles in degrees. Read the beam and angle definitions in
:doc:`../reduction/polarimetry` before applying it to observations.

.. tutorial-notebook:: polarimetry

.. tutorial-data::

Extract the shared archive and locate ``data/polarimetry.ecsv``. This is a
separate beam-measurement sequence, not the ordinary imaging stack.
The half-wave nominal fluxes are noiseless and their errors follow an illustrative
Poisson model; the quarter-wave table contains independent Poisson counts. They do not represent an instrumental polarization calibration.

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
contain no random noise. The fitter uses relative weights and scales parameter
covariance by the residual variance (``absolute_sigma=False``), so the returned
Stokes errors are nearly zero here despite nonzero beam errors. They are not
absolute photon-noise error estimates. A tiny fitted residual does not prove
the errors or instrument calibration are realistic. Inspect recovered Stokes uncertainties,
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
    :include-source: false
    :alt: Half-wave beam modulation with measurement error bars and a fitted curve; residuals are zero for the noiseless example.
    :caption: Recovered modulation and residuals for the eight beam pairs above. Error bars describe the supplied measurement model, not scatter injected into the nominal fluxes.

On observations, inspect the residual panel for outliers, angle-dependent
structure or unequal noise between positions. The noisy quarter-wave example
below shows residuals from an independent count realization.

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

Fit a noisy quarter-wave sequence with known calibration
--------------------------------------------------------

Read the separate quarter-wave sequence from the bundle. Unlike the first
example, its beam counts are independent Poisson draws. The normalization
``k=1.1`` and retarder zero ``7.5 deg`` are supplied instrument calibration
constants; the fit must estimate ``q``, ``u`` and ``v`` without using their
injected values. Keep only finite positive beam counts and uncertainties::

    >>> quarter = Table.read(data / 'polarimetry_quarterwave.ecsv')
    >>> valid = np.ones(len(quarter), dtype=bool)
    >>> for name in ('ordinary', 'extraordinary', 'ordinary_error', 'extraordinary_error'):
    ...     valid &= np.isfinite(quarter[name]) & (quarter[name] > 0)
    >>> quarter = quarter[valid]
    >>> quarter_psi = np.array(quarter['angle_deg'])
    >>> known_k, known_zero = 1.1, 7.5
    >>> phi = np.deg2rad(2 * (quarter_psi + known_zero))
    >>> basis = np.column_stack((np.cos(phi)**2, np.sin(phi) * np.cos(phi), -np.sin(phi)))
    >>> rank = np.linalg.matrix_rank(basis)
    >>> if len(quarter) <= 3 or rank < 3:
    ...     raise ValueError('Need redundant measurements that constrain all three Stokes components.')
    >>> print(f'Usable beam pairs: {len(quarter)}; independent model columns: {rank}')
    Usable beam pairs: 16; independent model columns: 3
    >>> quarter_solver = SLSDualBeamPolarimetry('quarterwave', k=known_k, zero=known_zero)
    >>> quarter_result = quarter_solver.compute(
    ...     quarter_psi, quarter['ordinary'], quarter['extraordinary'],
    ...     quarter['ordinary_error'], quarter['extraordinary_error'])
    >>> for name in ('q', 'u', 'v'):
    ...     parameter = getattr(quarter_result, name)
    ...     print(f'{name} = {parameter.nominal:+.4f} +/- {parameter.uncertainty:.4f}')
    q = +0.0404 +/- 0.0012
    u = -0.0330 +/- 0.0021
    v = +0.0196 +/- 0.0011
    >>> quarter_prediction = quarter_result.model(quarter_psi * u.deg)
    >>> quarter_residual = quarter_result.zi.nominal - quarter_prediction.nominal
    >>> print(f'Modulation residual standard deviation: {np.std(quarter_residual):.4f}')
    Modulation residual standard deviation: 0.0027

The independent-input error convention remains the same as for the half-wave
fit: the reported parameter errors use residual-scaled covariance, not an
absolute-noise calibration. The noise realization now produces nonzero
residuals and nonzero fitted errors. The recovered fractions can be compared
with the simulation's final check values, ``q=0.04``, ``u=-0.03``, ``v=0.02``;
noise prevents exact recovery. In observations those true values are unknown,
and calibration-constant uncertainty must also be considered.

.. plot:: _ext/workflow_figures.py plot_quarterwave
    :include-source: false
    :alt: Noisy quarter-wave beam modulation and fitted model across a full retarder cycle, with nonzero residuals and propagated beam-error bars.
    :caption: The shared plotting function uses the current quarter-wave fit for both the notebook and Sphinx replay. Calibration constants k and zero are fixed; none of the injected Stokes values enter the fit.

Save all three fitted Stokes components and their marginal standard errors,
with the supplied retarder calibration. This produces a separate product from
the half-wave result::

    >>> quarter_stokes = Table()
    >>> for name in ('q', 'u', 'v'):
    ...     parameter = getattr(quarter_result, name)
    ...     quarter_stokes[name] = [parameter.nominal]
    ...     quarter_stokes[name + '_error'] = [parameter.uncertainty]
    >>> quarter_stokes.meta.update({'retarder': 'quarterwave', 'beam_normalization': known_k,
    ...                             'zero_angle_deg': known_zero, 'input': 'polarimetry_quarterwave.ecsv',
    ...                             'error_convention': 'relative-weight residual-scaled marginal errors'})
    >>> quarter_stokes.write(output / 'stokes_quarterwave.ecsv', overwrite=True)
    >>> print('Saved:', 'stokes_quarterwave.ecsv')
    Saved: stokes_quarterwave.ecsv

Check angular coverage before fitting
-------------------------------------

Counting exposures alone does not check independent modulation information.
Repeated measurements at the same angle can reduce noise but do not add model
columns. For example, repeating one row of this design eight times still has
only one independent column::

    >>> repeated_basis = np.repeat(basis[:1], 8, axis=0)
    >>> print('Repeated-angle rank:', np.linalg.matrix_rank(repeated_basis), '; required:', 3)
    Repeated-angle rank: 1 ; required: 3

With fixed ``k`` and zero, missing quarter-wave angles can be acceptable when
the remaining design has full rank and adequate precision. Fitting zero adds
another parameter and needs an identifiability check beyond this three-column
linear design. Estimating normalization also requires usable measurements in
every phase of a cycle: four phases for half-wave and eight for quarter-wave.
Do not infer those requirements from optimizer convergence alone.

For example, retaining alternate half-wave positions leaves phases with no
measurement. Inspect the occupancy before requesting ``compute_k=True``::

    >>> phase = np.mod(np.rint(psi[::2] / 22.5).astype(int), 4)
    >>> print('Half-wave phase occupancy:', np.bincount(phase, minlength=4).tolist())
    Half-wave phase occupancy: [2, 0, 2, 0]

Restore missing phases, choose a validated externally measured ``k``, or reject
that sequence as unsuitable for the intended fit. The current implementation
can raise an internal ``TypeError`` when a normalization phase is entirely
absent; a populated phase containing only NaNs raises ``ValueError``. Precheck
coverage and finite fluxes rather than treating either exception as an
unpolarized-star result. See the module guide for normalization degeneracy,
low-polarization angle behavior and instrument-specific calibration limits.

.. doctest::
    :hide:

    >>> _temporary.cleanup()

.. raw:: html

    <span id="recover-known-half-wave-stokes-parameters"></span>
