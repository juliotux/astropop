Calibrate instrumental photometry with local standards
======================================================

Convert positive instrumental fluxes to magnitudes and estimate a zero point
from local standards in the same band. Reserve a check star and keep the target
out of the fit. This example fits only a zero point, not color or extinction.

.. tutorial-notebook:: photometric_calibration

.. tutorial-data::

Use the solved image and instrumental measurements from the preceding
stages, or a complete reference pair. Save calibrated magnitudes in ``reduced``.
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

Match measured stars to the standard catalog
---------------------------------------------

Use the WCS from :doc:`../astrometry_catalogs/astrometric_calibration` and the instrumental measurements
from :doc:`photometry`. Reference products allow starting here independently::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropy.table import Table
    >>> from astropop.framedata import read_framedata
    >>> from astropop.catalogs import SourcesCatalog
    >>> from astropop.math import QFloat

    >>> output = data.parent / 'reduced'
    >>> required = ('astrometric.fits', 'instrumental.ecsv')

    >>> # Require image and measurements from the same reduction, without mixing products.
    >>> available = [(output / name).is_file() for name in required]
    >>> if any(available) and not all(available):
    ...     raise ValueError('Incomplete reduced products: rerun the preceding tutorials together.')

    >>> inputs = output if all(available) else data / 'reference'

    >>> print('Input product set:', inputs.name)
    Input product set: reference

    >>> image = read_framedata(inputs / 'astrometric.fits')
    >>> measured = Table.read(inputs / 'instrumental.ecsv')
    >>> standards = Table.read(data / 'standards.ecsv')
    >>> if not len(measured) or not len(standards):
    ...     raise ValueError('No measurements or standards available for matching.')

    >>> catalog = SourcesCatalog(ids=standards['id'], ra=standards['ra'],
    ...                          dec=standards['dec'], unit='deg',
    ...                          mag={'R': QFloat(standards['R'], standards['R_error'])})
    >>> reference = catalog.get_coordinates()
    >>> reference_magnitudes = catalog.magnitude('R')
    >>> sky = image.wcs.pixel_to_world(measured['x'], measured['y'])
    >>> index, separation, _ = reference.match_to_catalog_sky(sky)

    >>> # Reject standards that select the same measured detection.
    >>> unique = np.bincount(index, minlength=len(measured))[index] == 1
    >>> flux, flux_error = measured['flux'][index], measured['flux_error'][index]
    >>> usable = ((separation < 0.5 * u.arcsec) & (measured['flags'][index] == 0)
    ...           & unique & np.isfinite(flux) & np.isfinite(flux_error)
    ...           & (flux > 0) & (flux_error > 0))
    >>> reference_valid = (np.isfinite(reference_magnitudes.nominal)
    ...                    & np.isfinite(reference_magnitudes.uncertainty)
    ...                    & (reference_magnitudes.uncertainty > 0))
    >>> calibrators = usable & reference_valid & standards['standard']
    >>> check = usable & reference_valid & ~standards['standard'] & (standards['id'] == 'S09')
    >>> if calibrators.sum() < 2 or not np.any(check):
    ...     raise ValueError('This workflow needs two standards and the withheld check star.')

    >>> print(f'Matched sources: {usable.sum()}; zero-point standards: {calibrators.sum()}')
    Matched sources: 9; zero-point standards: 7

Use the image and table from one product set. If only one required product
exists in ``reduced``, the example stops instead of mixing it with a reference
file. Rerun the preceding stages together, or explicitly choose
``inputs = data / 'reference'`` before loading both files to use the bundled
example. This presence check cannot establish freshness: after changing a crop,
registration reference or photometry settings, regenerate all downstream
products even if files with the expected names already exist.

SourcesCatalog keeps identifiers, positions and uncertain R magnitudes in one
interface; ``magnitude`` returns a QFloat whose uncertainty is the catalog
standard error. The zero-point calculation below uses those numerical values
explicitly so its weighting and covariance assumptions remain visible.

This simulation has known, isolated standards. The standard magnitudes refer
to its synthetic R band; they do not define a transformation to an actual
instrument's passband. The selection rejects all associations that share a
nearest detection with another catalog row; it does not resolve crowded-field
ambiguities. It excludes flagged measurements and nonfinite or nonpositive
fluxes and flux errors. Only standards and check stars need valid reference
magnitudes and positive reference errors; a target can be calibrated without
a reference magnitude.

Requiring two standards and the withheld check star is this tutorial's policy,
not an Astropop API restriction. Inspect the matches and exclude variable or
saturated standards before fitting observations.

Fit a zero point with an explicit exposure convention
------------------------------------------------------

The combined image is a mean on a 30-second exposure's ADU scale. Compute
``m_inst = -2.5*log10(flux/exposure)`` and propagate flux errors with
``sigma_m = (2.5/ln(10))*sigma_flux/flux``. Treat exposure duration as exact here::

    >>> exposure = image.meta['EXPTIME']
    >>> if not np.isfinite(exposure) or exposure <= 0:
    ...     raise ValueError('EXPTIME must be finite and positive.')

    >>> instrumental = np.full(len(standards), np.nan)
    >>> instrumental_error = np.full(len(standards), np.nan)

    >>> # Divide by exposure duration to define instrumental magnitudes from count rate.
    >>> instrumental[usable] = -2.5 * np.log10(flux[usable] / exposure)

    >>> # The logarithm's derivative converts relative flux error to magnitude error.
    >>> instrumental_error[usable] = 2.5 / np.log(10) * flux_error[usable] / flux[usable]
    >>> offsets = reference_magnitudes.nominal - instrumental
    >>> offset_error = np.hypot(reference_magnitudes.uncertainty, instrumental_error)

    >>> # Fit the zero point using inverse-variance weights of independent offsets.
    >>> weights = 1 / np.square(offset_error[calibrators])
    >>> zero_point = np.sum(weights * offsets[calibrators]) / np.sum(weights)
    >>> zero_point_error = 1 / np.sqrt(np.sum(weights))
    >>> calibrated = instrumental + zero_point

    >>> # This simple error budget omits covariance between the fit and its calibrators.
    >>> calibrated_error = np.hypot(instrumental_error, zero_point_error)

    >>> print(f'Zero point: {zero_point:.3f} +/- {zero_point_error:.3f} mag')
    Zero point: 25.000 +/- 0.005 mag
    >>> print(f'Check-star residual: {np.asarray(calibrated - standards["R"])[check][0]:+.3f} mag')
    Check-star residual: +0.009 mag

.. plot:: _ext/workflow_figures.py plot_photometric_calibration
    :include-source: false
    :alt: Individual standard-star zero points and calibrated-minus-reference residuals versus reference magnitude, with a separate check star.
    :caption: The fitted zero point is compared with individual standards and an independent check star. The injected zero point is 25 mag; finite apertures and noise prevent exact recovery.

Only usable measurements enter the logarithm; rejected rows retain NaN.
Inspect individual offsets before accepting the fit. An isolated outlier calls
for checking its association, flags and variability; a trend with magnitude or
color suggests a measurement or model problem. Do not automatically clip these
patterns away. After excluding a standard for a documented reason, recompute
the fit and check-star residual.

The zero-point uncertainty above assumes independent catalog and instrumental
errors and a correct constant-offset model. Standards used in the fit share
covariance with that zero point, so their residual/error bars are diagnostics,
not independent goodness-of-fit measurements. The displayed and saved
``R_error`` values add instrumental and zero-point variances as if independent;
for fitted standards this omits their covariance with the zero point.
Reusing the zero point also
correlates calibrated magnitudes. Shared calibration errors, aperture corrections,
passband/color terms and atmospheric extinction are not included. Do not add
extra fitted terms unless the observations constrain them.

Save calibrated measurements and provenance
--------------------------------------------

Keep units, the standard release/band, exposure convention and fitted calibration
with the table. Magnitudes without a zero-point convention are ambiguous::

    >>> result = Table({'id': standards['id'][usable], 'R': calibrated[usable],
    ...                 'R_error': calibrated_error[usable],
    ...                 'used_in_fit': calibrators[usable]})
    >>> result['R'].unit = result['R_error'].unit = u.mag
    >>> result.meta.update({'band': 'synthetic R', 'catalog': 'bundled synthetic standards',
    ...                     'input_set': inputs.name, 'image': 'astrometric.fits',
    ...                     'measurements': 'instrumental.ecsv',
    ...                     'zero_point': float(zero_point),
    ...                     'zero_point_error': float(zero_point_error),
    ...                     'error_convention': 'independent errors; fit covariance omitted',
    ...                     'flux_convention': 'ADU per second; mean of 30-second frames'})
    >>> output.mkdir(exist_ok=True)
    >>> result.write(output / 'calibrated_photometry.ecsv', overwrite=True)

    >>> print(f'Saved calibrated sources: {len(result)}')
    Saved calibrated sources: 9

Compare an independent check star and repeat observations before trusting
absolute calibration. This local workflow makes no live catalog query.

.. doctest::
    :hide:

    >>> _temporary.cleanup()
