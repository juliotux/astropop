Recover linear and circular polarization from quarter-wave measurements
=======================================================================

This independent sequence starts with a noisy beam table, supplied normalization
and retarder-zero calibration. It does not reuse the half-wave images or fit.

.. tutorial-data::

.. tutorial-notebook:: quarterwave

Run from the extracted bundle root:

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

Prepare the fitting interfaces and output directory::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropy.table import Table
    >>> from astropop.polarimetry import SLSDualBeamPolarimetry

    >>> output = data.parent / 'reduced'
    >>> output.mkdir(exist_ok=True)

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

    >>> # The three columns multiply q, u and v; their rank checks angle coverage.
    >>> basis = np.column_stack((np.cos(phi)**2, np.sin(phi) * np.cos(phi), -np.sin(phi)))
    >>> rank = np.linalg.matrix_rank(basis)
    >>> if len(quarter) <= 3 or rank < 3:
    ...     raise ValueError('Need redundant measurements that constrain all three Stokes components.')

    >>> print(f'Usable beam pairs: {len(quarter)}; independent model columns: {rank}')
    Usable beam pairs: 16; independent model columns: 3

    >>> # Treat the independently calibrated beam normalization and zero angle as fixed.
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

    >>> # Repeating one angle adds measurements without adding independent constraints.
    >>> repeated_basis = np.repeat(basis[:1], 8, axis=0)

    >>> print('Repeated-angle rank:', np.linalg.matrix_rank(repeated_basis), '; required:', 3)
    Repeated-angle rank: 1 ; required: 3

With fixed ``k`` and zero, missing quarter-wave angles can be acceptable when
the remaining design has full rank and adequate precision. Fitting zero adds
another parameter and needs an identifiability check beyond this three-column
linear design. Estimating normalization also requires usable measurements in
every phase of a cycle: four phases for half-wave and eight for quarter-wave.
Do not infer those requirements from optimizer convergence alone.

The half-wave tutorial separately demonstrates phase occupancy for its
normalization estimator.

.. doctest::
    :hide:

    >>> _temporary.cleanup()
