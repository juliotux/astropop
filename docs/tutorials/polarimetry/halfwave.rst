Recover Stokes parameters from paired beam fluxes
=================================================

Measure paired beams in eight ideal half-wave images and recover their linear
polarization. For an independent linear/circular-polarization sequence, see
:doc:`quarterwave`.
The local images require no online catalog. This workflow uses a half-wave
retarder and unit beam normalization, with matched ordinary/extraordinary
beams and angles in degrees. Read the beam and angle definitions in
:doc:`../../modules/polarimetry/index` before applying it to observations.

.. tutorial-notebook:: polarimetry

.. tutorial-data::

Extract the shared archive and locate ``data/polarimetry/images`` and
``data/polarimetry_quarterwave.ecsv``. These are separate polarimetric sequences,
not the ordinary imaging stack. The half-wave images contain ideal Gaussian
beams on a constant sky of 100 electrons per pixel, with no random noise or
detector defects. They represent already calibrated images; real images need
bias/flat correction and quality masks before measurement. The quarter-wave
table contains independent Poisson counts. Neither supplies an instrumental
polarization calibration.

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

Detect, pair and measure the beams
----------------------------------

The lower-left beam is ordinary in this simulation, and its extraordinary
partner is displaced by ``(+24,+24)`` pixels. That physical assignment and
separation are supplied instrument information. ``match_pairs`` identifies
geometric neighbors; it cannot determine which physical beam is ordinary.

Read the first calibrated image and detect its two sources::

    >>> import numpy as np
    >>> from astropy.table import Table
    >>> from astropop.framedata import read_framedata
    >>> from astropop.photometry import daofind, aperture_photometry
    >>> from astropop.polarimetry import match_pairs, SLSDualBeamPolarimetry

    >>> image_files = sorted((data / 'polarimetry' / 'images').glob('halfwave_*.fits'))
    >>> first_image = read_framedata(image_files[0])
    >>> detections = daofind(first_image.data, threshold=5, background=100.,
    ...                      noise=10., fwhm=3.53)

    >>> # The displacement uses ordinary minus extraordinary coordinates.
    >>> pairs = match_pairs(detections['x'], detections['y'], -24., -24., tolerance=2.)

    >>> print(f'Detected beams: {len(detections)}; geometric pairs: {len(pairs)}')
    Detected beams: 2; geometric pairs: 1

    >>> pair = pairs[0]

The negative displacement follows ``match_pairs``'s ``(x-dx,y-dy)`` convention.
The images have no drift, so the first image's measured centroids can be reused.
Use an annulus to subtract sky from each beam; supply its counting-noise standard
deviation separately from the source Poisson term::

    >>> rows = []
    >>> for filename in image_files:
    ...     frame = read_framedata(filename)
    ...     measured = aperture_photometry(frame.data, detections['x'], detections['y'],
    ...                                    r=6, r_ann=(9, 12), gain=1., bkg_error=10.,
    ...                                    mask=frame.mask, pixel_flags=frame.flags)
    ...     if np.any(measured['flags'] != 0):
    ...         raise ValueError('Inspect flagged beam measurements before fitting.')
    ...
    ...     # Reuse the geometric pair's row indices to keep the beam labels consistent.
    ...     ordinary_row, extraordinary_row = measured[pair['o']], measured[pair['e']]
    ...     rows.append((frame.meta['RETANGLE'], ordinary_row['flux'],
    ...                  extraordinary_row['flux'], ordinary_row['flux_error'],
    ...                  extraordinary_row['flux_error']))

    >>> beams = Table(rows=rows, names=['angle_deg', 'ordinary', 'extraordinary',
    ...                                'ordinary_error', 'extraordinary_error'])
    >>> beams.meta.update({'flux_unit': 'electron', 'aperture_radius_pixels': 6,
    ...                    'sky_annulus_pixels': [9, 12]})
    >>> psi = np.array(beams['angle_deg'])
    >>> ordinary, extraordinary = beams['ordinary'], beams['extraordinary']

    >>> print(f'Beam pairs: {len(beams)}; angle spacing: {psi[1] - psi[0]:.1f} deg')
    Beam pairs: 8; angle spacing: 22.5 deg

A finite aperture loses the same profile fraction from both identical beams
here, so their normalized difference is preserved. Unequal PSFs, a truncated
aperture or contamination can instead bias it. The nominal images are noiseless;
the nonzero errors describe an illustrative source-plus-sky counting model,
not measured scatter in these images.

.. plot:: _ext/workflow_figures.py plot_beam_image
    :include-source: false
    :alt: An ideal half-wave image with two detected beams, their source apertures and sky annuli, labeled ordinary and extraordinary.
    :caption: The measured positions and supplied physical beam assignment used above. Matching uses the calibrated separation; photometry measures the original pixels at those positions.

Fit and inspect the polarization
---------------------------------

Use ``'halfwave'`` for a half-wave retarder, ``k=1`` for already normalized
beams and ``zero=0`` for a known zero angle of zero degrees. Supply a standard
uncertainty for each beam measurement::

    >>> # Supply the known beam normalization and zero angle instead of fitting them.
    >>> solver = SLSDualBeamPolarimetry('halfwave', k=1., zero=0.)
    >>> result = solver.compute(psi, ordinary, extraordinary,
    ...                         beams['ordinary_error'], beams['extraordinary_error'])

    >>> print(f'q = {result.q.nominal:.3f}, u = {result.u.nominal:.3f}')
    q = 0.030, u = -0.020
    >>> print(f'Linear polarization: {100 * result.p.nominal:.2f}%')
    Linear polarization: 3.61%
    >>> print(f'Position angle: {result.theta.nominal:.2f} {result.theta.unit}')
    Position angle: 163.15 deg

Here the supplied errors are an illustrative source-plus-sky Poisson model and the nominal data
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

    >>> # Pass explicit angular units when evaluating the fitted modulation.
    >>> predicted = result.model(psi * u.deg)
    >>> residual = result.zi.nominal - predicted.nominal

    >>> print(f'Largest modulation residual: {np.max(np.abs(residual)):.3f}')
    Largest modulation residual: 0.000

.. plot:: _ext/workflow_figures.py plot_polarimetry
    :include-source: false
    :alt: Half-wave beam modulation with measurement error bars and a fitted curve; residuals are zero for the noiseless example.
    :caption: Recovered modulation and residuals for the eight beam pairs above. Error bars describe the supplied measurement model, not scatter injected into the nominal fluxes.

On observations, inspect the residual panel for outliers, angle-dependent
structure or unequal noise between positions. The :doc:`quarterwave` tutorial shows residuals from an independent count
realization.

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
    >>> beams.write(output / 'beam_photometry.ecsv', overwrite=True)
    >>> stokes.write(output / 'stokes.ecsv', overwrite=True)

    >>> print(f'Saved Stokes measurements: {len(stokes)}')
    Saved Stokes measurements: 1

``beam_photometry.ecsv`` also retains the extracted fluxes and measurement errors,
so the fit can be repeated without remeasuring the images.

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


For example, retaining alternate half-wave positions leaves phases with no
measurement. Inspect the occupancy before requesting ``compute_k=True``::

    >>> # Sampling every other exposure misses two of the four half-wave phases.
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
