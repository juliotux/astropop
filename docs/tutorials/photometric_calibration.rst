Calibrate instrumental photometry with local standards
======================================================

Convert positive instrumental fluxes to magnitudes and estimate a zero point
from local standards in the same band. Reserve a check star and keep the target
out of the fit. This example fits only a zero point, not color or extinction.

.. tutorial-notebook:: photometric_calibration

.. tutorial-data::

Extract the shared archive once. Its ``data/reference`` directory lets you
start this stage independently; earlier stages save your own products in
``reduced``. Run the notebook from the extracted bundle or set the data path
when following this page::

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

Use the WCS from :doc:`astrometric_calibration` and the instrumental measurements
from :doc:`photometry`. Reference products allow starting here independently::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy.table import Table
    >>> from astropop.framedata import read_framedata
    >>> output = data.parent / 'reduced'
    >>> required = ('astrometric.fits', 'instrumental.ecsv')
    >>> available = [(output / name).is_file() for name in required]
    >>> if any(available) and not all(available):
    ...     raise ValueError('Incomplete reduced products: rerun the preceding tutorials together.')
    >>> inputs = output if all(available) else data / 'reference'
    >>> print('Input product set:', inputs.name)
    Input product set: reference
    >>> image = read_framedata(inputs / 'astrometric.fits')
    >>> measured = Table.read(inputs / 'instrumental.ecsv')
    >>> standards = Table.read(data / 'standards.ecsv')
    >>> reference = SkyCoord(standards['ra'], standards['dec'])
    >>> sky = image.wcs.pixel_to_world(measured['x'], measured['y'])
    >>> index, separation, _ = reference.match_to_catalog_sky(sky)
    >>> flux, flux_error = measured['flux'][index], measured['flux_error'][index]
    >>> usable = ((separation < 0.5 * u.arcsec) & (measured['flags'][index] == 0)
    ...           & np.isfinite(flux) & np.isfinite(flux_error) & (flux > 0))
    >>> calibrators = usable & standards['standard']
    >>> print(f'Matched sources: {usable.sum()}; zero-point standards: {calibrators.sum()}')
    Matched sources: 9; zero-point standards: 7

Use the image and table from one product set. If only one required product
exists in ``reduced``, the example stops instead of mixing it with a reference
file. Rerun the preceding stages together, or explicitly choose
``inputs = data / 'reference'`` before loading both files to use the bundled
example. This presence check cannot establish freshness: after changing a crop,
registration reference or photometry settings, regenerate all downstream
products even if files with the expected names already exist.

This simulation has known, isolated standards. The standard magnitudes refer
to its synthetic R band; they do not define a transformation to an actual
instrument's passband. Resolve duplicate associations, exclude variable or
saturated standards, and inspect magnitude/color ranges on observations.

Fit a zero point with an explicit exposure convention
------------------------------------------------------

The combined image is a mean on a 30-second exposure's ADU scale. Compute
``m_inst = -2.5*log10(flux/exposure)`` and propagate flux errors with
``sigma_m = (2.5/ln(10))*sigma_flux/flux``. Treat exposure duration as exact here::

    >>> exposure = image.meta['EXPTIME']
    >>> instrumental = -2.5 * np.log10(flux / exposure)
    >>> instrumental_error = 2.5 / np.log(10) * flux_error / flux
    >>> offsets = standards['R'] - instrumental
    >>> offset_error = np.hypot(standards['R_error'], instrumental_error)
    >>> weights = 1 / np.square(offset_error[calibrators])
    >>> zero_point = np.sum(weights * offsets[calibrators]) / np.sum(weights)
    >>> zero_point_error = 1 / np.sqrt(np.sum(weights))
    >>> calibrated = instrumental + zero_point
    >>> calibrated_error = np.hypot(instrumental_error, zero_point_error)
    >>> print(f'Zero point: {zero_point:.3f} +/- {zero_point_error:.3f} mag')
    Zero point: 25.000 +/- 0.005 mag
    >>> check = usable & (standards['id'] == 'S09')
    >>> print(f'Check-star residual: {np.asarray(calibrated - standards["R"])[check][0]:+.3f} mag')
    Check-star residual: +0.009 mag

.. plot:: _ext/workflow_figures.py plot_photometric_calibration
    :include-source: False
    :alt: Individual standard-star zero points and calibrated-minus-reference residuals versus reference magnitude, with a separate check star.
    :caption: The fitted zero point is compared with individual standards and an independent check star. The injected zero point is 25 mag; finite apertures and noise prevent exact recovery.

The zero-point uncertainty above assumes independent catalog and instrumental
errors and a correct constant-offset model. Standards used in the fit share
covariance with that zero point, so their residual/error bars are diagnostics,
not independent goodness-of-fit measurements. Reusing the zero point also
correlates calibrated magnitudes. Shared calibration errors, aperture corrections,
passband/color terms and atmospheric extinction are not included. Do not add
extra fitted terms unless the observations constrain them.

Save calibrated measurements and provenance
--------------------------------------------

Keep units, the standard release/band, exposure convention and fitted calibration
with the table. Magnitudes without a zero-point convention are ambiguous::

    >>> result = Table({'id': standards['id'][usable], 'R': calibrated[usable],
    ...                 'R_error': calibrated_error[usable]})
    >>> result['R'].unit = result['R_error'].unit = u.mag
    >>> result.meta.update({'band': 'synthetic R', 'catalog': 'bundled synthetic standards',
    ...                     'input_set': inputs.name, 'image': 'astrometric.fits',
    ...                     'measurements': 'instrumental.ecsv',
    ...                     'zero_point': float(zero_point),
    ...                     'zero_point_error': float(zero_point_error),
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
