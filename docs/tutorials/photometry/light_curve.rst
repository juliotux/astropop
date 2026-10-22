Measure a differential light curve
==================================

Measure the same stars on individual calibrated exposures. Stacking would erase
the variation we want to follow. This independent twelve-exposure sequence has
one variable target, three comparison stars and small pointing shifts.

.. tutorial-notebook:: light_curve

.. tutorial-data::

Use the separate exposures in ``data/time_series/raw`` and the bias/flat
masters. This stage writes individual measurements and a differential light curve.
See :doc:`../index` for archive layout, configuration and rerun policy.
Run from the extracted bundle root when following the HTML examples::

    >>> from pathlib import Path

.. code-block:: python

    data = Path('data')

.. doctest::
    :hide:

    >>> from tempfile import TemporaryDirectory
    >>> from tutorial_data import generate_dataset

    >>> _temporary = TemporaryDirectory()
    >>> data = Path(_temporary.name).resolve() / 'data'
    >>> generate_dataset(data, include_extras=True)

Calibrate the time series and identify stars
--------------------------------------------

Use the master bias and flat from the earlier stages or their reference pair.
The finder table provides rounded initial positions and source roles, not exact
injected centroids. Refine those positions using detections on the first frame::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropy.table import Table
    >>> from astropy.time import Time
    >>> from astropop.framedata import read_framedata
    >>> from astropop.image.processing import subtract_bias, flat_correct
    >>> from astropop.image.register import register_framedata_list
    >>> from astropop.photometry import background, daofind, aperture_photometry

    >>> output = data.parent / 'reduced'
    >>> own_masters = all((output / name).exists() for name in ['master_bias.fits', 'master_flat.fits'])
    >>> inputs = output if own_masters else data / 'reference'
    >>> bias = read_framedata(inputs / 'master_bias.fits')
    >>> flat = read_framedata(inputs / 'master_flat.fits')
    >>> finder = Table.read(data / 'time_series' / 'finder.ecsv')
    >>> frames = []
    >>> for path in sorted((data / 'time_series' / 'raw').glob('science_*.fits')):
    ...     frame = read_framedata(path)
    ...
    ...     # At gain = 1, combine clipped signal and squared read noise in ADU variance.
    ...     frame.uncertainty = np.sqrt(np.maximum(frame.data - bias.data, 0)
    ...                                 + frame.meta['RDNOISE']**2)
    ...     frames.append(flat_correct(subtract_bias(frame, bias), flat))

    >>> sky0, rms0 = background(frames[0].data, box_size=32, global_bkg=False)
    >>> detected = daofind(frames[0].data, threshold=5, background=sky0, noise=rms0, fwhm=4.24)
    >>> distance2 = ((detected['x'][:, None] - finder['x'])**2
    ...              + (detected['y'][:, None] - finder['y'])**2)

    >>> # Each finder-list source selects its nearest first-frame detection.
    >>> nearest = np.argmin(distance2, axis=0)
    >>> x0, y0 = detected['x'][nearest], detected['y'][nearest]

    >>> print('Source roles:', finder['role'].tolist())
    Source roles: ['target', 'comparison', 'comparison', 'comparison']

These isolated stars have unique nearest detections. Inspect associations and
use a distance limit for crowded or uncertain finder coordinates. The target
is S01; it must not contribute to the comparison ensemble. The simulated
masters have no defects; real masks require the policy in
:doc:`../diagnostics/quality_propagation`.

Follow coordinates without interpolating the measured pixels
-------------------------------------------------------------

Obtain transforms with the registration API, but measure the original calibrated
frames. For translations, add the reported reference-to-moving shifts to the
reference aperture positions. This avoids resampling the photometry pixels::

    >>> registered = register_framedata_list(frames, ref_image=0, upsample_factor=20,
    ...                                        window=(slice(32, 224), slice(32, 224)))
    >>> fluxes, errors, times, flags = [], [], [], []
    >>> for frame, shifted in zip(frames, registered):
    ...     dx = shifted.meta['astropop registration_shift_x']
    ...     dy = shifted.meta['astropop registration_shift_y']
    ...     sky, rms = background(frame.data, box_size=32, global_bkg=False)
    ...
    ...     # Measure original calibrated pixels; add reference-to-moving shifts to positions.
    ...     measured = aperture_photometry(frame.data, x0 + dx, y0 + dy, r=6.,
    ...                                    r_ann=(9., 13.), bkg_error=rms, gain=1.,
    ...                                    mask=frame.mask, pixel_flags=frame.flags)
    ...     fluxes.append(measured['flux'])
    ...     errors.append(measured['flux_error'])
    ...     flags.append(measured['flags'])
    ...
    ...     # Use the exposure midpoint in UTC; this is not a barycentric time correction.
    ...     start = Time(frame.meta['DATE-OBS'], format='isot', scale='utc')
    ...     times.append((start + frame.meta['EXPTIME'] / 2 * u.s).jd)

    >>> fluxes, errors, flags = np.array(fluxes), np.array(errors), np.array(flags)

    >>> # Keep exposures with a valid target and the same complete comparison ensemble.
    >>> usable = ((flags == 0).all(axis=1) & np.isfinite(fluxes).all(axis=1)
    ...           & np.isfinite(errors).all(axis=1) & (fluxes > 0).all(axis=1)
    ...           & (errors > 0).all(axis=1))
    >>> if not np.any(usable):
    ...     raise ValueError('No exposure has valid target and comparison measurements.')

    >>> print(f'Usable exposures: {usable.sum()} / {len(frames)}')
    Usable exposures: 12 / 12

This coordinate rule is specific to translations. Use the complete transform
for rotation/scale, and inspect registration failures before measuring. The
metadata specify exposure-start UTC here; adding half the exposure gives
mid-exposure JD in UTC. This is not a barycentric time correction. For timing
work requiring BJD in TDB, the correction also needs the observatory location
and target sky position; changing a column name or time scale alone does not
apply the light-travel correction.

The input noise estimate uses the header's read noise and assumes this dataset's
gain of one electron per ADU. For another gain, convert the Poisson and read
noise terms to the image's units before combining them. Fixed apertures are
suitable here because the simulated PSF width is constant. On observations,
inspect widths and centering across the sequence, especially near bad pixels
or when seeing changes.

Normalize a comparison ensemble
--------------------------------

Normalize each comparison star by its own median flux, then average those
relative curves. Divide the target's relative flux by that ensemble. Propagate
independent per-exposure measurement errors through the division::

    >>> flux, error = fluxes[usable], errors[usable]

    >>> # Column zero is the target; normalize each comparison by its own median.
    >>> comparison_levels = np.median(flux[:, 1:], axis=0)
    >>> comparisons = flux[:, 1:] / comparison_levels
    >>> comparison_count = comparisons.shape[1]
    >>> ensemble = comparisons.mean(axis=1)

    >>> # Combine independent errors, treating the normalization medians as fixed.
    >>> ensemble_error = np.sqrt(((error[:, 1:] / comparison_levels)**2).sum(axis=1)) / comparison_count

    >>> # Normalize the target separately before removing the ensemble's common variation.
    >>> target = flux[:, 0] / np.median(flux[:, 0])
    >>> differential = target / ensemble
    >>> differential_error = differential * np.sqrt((error[:, 0] / flux[:, 0])**2
    ...                                              + (ensemble_error / ensemble)**2)
    >>> curve = Table({'jd_utc': np.array(times)[usable], 'target_relative': target,
    ...                'ensemble': ensemble, 'relative_flux': differential,
    ...                'flux_error': differential_error})
    >>> curve['elapsed_min'] = (curve['jd_utc'] - curve['jd_utc'][0]) * 24 * 60
    >>> output.mkdir(exist_ok=True)
    >>> curve.write(output / 'light_curve.ecsv', overwrite=True)

    >>> # Preserve every measurement and its flags, including rejected exposures.
    >>> individual = Table({'jd_utc': np.repeat(times, len(finder)),
    ...                     'id': np.tile(finder['id'], len(frames)),
    ...                     'flux': fluxes.ravel(), 'flux_error': errors.ravel(),
    ...                     'flags': flags.ravel()})
    >>> individual.write(output / 'time_series_photometry.ecsv', overwrite=True)

    >>> print(f'Differential flux range: {differential.min():.2f} to {differential.max():.2f}')
    Differential flux range: 0.87 to 1.19

    >>> truth = Table.read(data / 'time_series' / 'truth.ecsv')

    >>> # Use simulation truth only to inspect recovery, not to calibrate the curve.
    >>> difference = differential - truth['relative_flux'][usable]

    >>> print(f'Largest simulation difference: {100 * np.max(np.abs(difference)):.1f}%')
    Largest simulation difference: 4.5%

.. plot:: _ext/workflow_figures.py plot_light_curve
    :include-source: false
    :alt: Target and comparison ensemble flux versus time, followed by a differential light curve with measurement errors and injected variability.
    :caption: The comparison ensemble follows shared throughput changes. Dividing by it recovers the target's variability. Injected truth is used only for this final simulation comparison.

The point near three minutes is high relative to the injected curve; inspect
its individual measurement and comparison stars before interpreting such a
point as an additional event. Finite photon/background noise and uncertain
normalization prevent exact recovery of the injected amplitude.

The per-source table retains measurements and flags even for exposures rejected
from the differential curve. The strict selection here requires all three
comparison stars and the target to be usable in the same exposure. A missing,
flagged or invalid comparison rejects that exposure. Keeping the same ensemble
avoids introducing offsets by silently changing its membership. In this finder
table the target is the first row and the comparisons are the remaining rows;
preserve that ordering when adapting the example.

Inspect the individual comparison stars
---------------------------------------

A stable ensemble average can hide a discrepant comparison star. First plot
each comparison divided by its own median, then divide it by the mean of the
**other** comparison stars. Excluding the star under inspection from its
reference avoids suppressing its deviations through self-inclusion. This
diagnostic requires at least two comparison stars; it uses the same fixed
membership and accepted exposures as the target curve.

.. plot:: _ext/workflow_figures.py plot_comparison_curves
    :include-source: false
    :alt: Each comparison star's normalized flux follows the shared throughput trend; a second panel divides each star by the ensemble of the other comparisons to reveal relative deviations.
    :caption: The shared workflow helper uses the current notebook measurements for both panels. Individual ratios reveal differences that averaging could hide. The panels show measured curves, without using injected target variability as a calibration input.

Shared throughput changes should appear together in the upper panel. In the
lower panel, investigate a discrepant measurement or persistent trend by
inspecting its aperture, sky annulus, masks and centering. Deviations include
measurement noise; they do not by themselves prove a star is variable. A bad
comparison also affects the reference of the remaining stars. Try physically
justified subsets and remeasure the target before interpreting a small signal.

The formal errors above treat the median normalization levels as fixed and omit
shared-master errors and correlations between comparison stars. They are not a
complete time-series noise model. Inspect individual comparison curves and
repeat the analysis with different reference subsets before interpreting a small
signal. This short sequence illustrates relative photometry; it does not fit a
period or calibrate absolute magnitudes. Continue with
:doc:`photometric_calibration` for an instrumental-to-standard zero point.

.. doctest::
    :hide:

    >>> _temporary.cleanup()
