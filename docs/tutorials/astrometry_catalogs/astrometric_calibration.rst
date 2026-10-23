Fit an astrometric solution from local references
=================================================

Fit a celestial WCS to measured centroids and a local sky catalog. This is an
offline refinement with approximate pointing, not blind field identification.
Seven reference stars constrain the fit; two others check it independently.

.. tutorial-notebook:: astrometric_calibration

.. tutorial-data::

Use the combined image and instrumental table from :doc:`../photometry/photometry`, or a
complete reference pair. This stage saves an image with the fitted WCS and
a table of reference matches in ``reduced``.
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

Associate detections with local reference stars
-----------------------------------------------

Use the instrumental source table from :doc:`../photometry/photometry` when available, or its
reference equivalent. The pointing file provides an approximate TAN projection,
with a deliberate pixel offset. It is not the solution we are trying to recover::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropy.io import fits
    >>> from astropy.table import Table
    >>> from astropy.wcs import WCS
    >>> from astropy.wcs.utils import fit_wcs_from_points
    >>> from astropop.framedata import read_framedata
    >>> from astropop.catalogs import SourcesCatalog

    >>> output = data.parent / 'reduced'
    >>> required = ('combined.fits', 'instrumental.ecsv')
    >>> available = [(output / name).is_file() for name in required]
    >>> if any(available) and not all(available):
    ...     raise ValueError('Incomplete reduced products: rerun the preceding tutorials together.')

    >>> inputs = output if all(available) else data / 'reference'

    >>> print('Input product set:', inputs.name)
    Input product set: reference

    >>> measured = Table.read(inputs / 'instrumental.ecsv')
    >>> standards = Table.read(data / 'standards.ecsv')
    >>> catalog = SourcesCatalog(ids=standards['id'], ra=standards['ra'],
    ...                          dec=standards['dec'], unit='deg')
    >>> reference = catalog.get_coordinates()
    >>> rough = WCS(fits.getheader(data / 'pointing.fits'))

    >>> # The rough pointing WCS is only used to associate detections with standards.
    >>> x_guess, y_guess = rough.world_to_pixel(reference)
    >>> distance2 = ((measured['x'][:, None] - x_guess)**2
    ...              + (measured['y'][:, None] - y_guess)**2)
    >>> nearest = np.argmin(distance2, axis=0)
    >>> separation_pixels = np.sqrt(distance2[nearest, np.arange(len(reference))])
    >>> usable = (separation_pixels < 8) & (measured['flags'][nearest] == 0)

    >>> print(f'References within 8 pixels: {usable.sum()} / {len(reference)}')
    References within 8 pixels: 9 / 9

Use the image and table from one product set. If only one required product
exists in ``reduced``, the example stops instead of mixing it with a reference
file. Rerun the preceding stages together, or explicitly choose
``inputs = data / 'reference'`` before loading both files to use the bundled
example. This presence check cannot establish freshness: after changing a crop,
registration reference or photometry settings, regenerate all downstream
products even if files with the expected names already exist.

SourcesCatalog supplies identified sky positions; the measured centroids come
from the preceding photometry stage. The catalog does not fit the WCS.

The tolerance is in detector pixels because this association starts from an
approximate WCS. The references are isolated and uniquely associated here;
independent nearest matches do not enforce a one-to-one assignment. Inspect the
:doc:`local_catalog_matching` diagrams and resolve ambiguous matches on real
fields. Use compatible coordinate frames and observation epochs before fitting.

Fit and check the projection
-----------------------------

Use seven references distributed across the field. Keep the last two out of the
fit so their residuals provide a separate check. Pixel positions use zero-based
``(x, y)`` order, while image arrays use ``[y, x]``::

    >>> # Reserve other matched stars to check the solution independently of the fit.
    >>> fit_rows = usable & (np.arange(len(reference)) < 7)
    >>> check_rows = usable & ~fit_rows
    >>> x, y = measured['x'][nearest], measured['y'][nearest]
    >>> solution = fit_wcs_from_points((x[fit_rows], y[fit_rows]), reference[fit_rows],
    ...                                projection='TAN')
    >>> recovered = solution.pixel_to_world(x, y)

    >>> # Use spherical offsets so the east residual includes the declination factor.
    >>> east, north = reference.spherical_offsets_to(recovered)
    >>> radial = np.hypot(east.arcsec, north.arcsec)

    >>> print(f'Fit stars: {fit_rows.sum()}; independent check stars: {check_rows.sum()}')
    Fit stars: 7; independent check stars: 2
    >>> print(f'Check-star RMS: {np.sqrt(np.mean(radial[check_rows]**2)):.3f} arcsec')
    Check-star RMS: 0.004 arcsec

.. plot:: _ext/workflow_figures.py plot_astrometry
    :include-source: false
    :alt: Reference positions across the image and east/north WCS residuals, distinguishing fit stars from withheld check stars.
    :caption: Small fit residuals alone are insufficient; compare the withheld stars and look for spatial structure. This synthetic field follows a TAN projection without optical distortion.

Astropy performs the TAN fit in this example; Astropop supplies image handling
and the measured source table. A translation-only image registration is not an
astrometric calibration. Real distortion may require another model and more
reference stars. Two check stars are only a small diagnostic sample, not a
complete validation of accuracy across a real field. The simulated reference
positions are exact; real catalog errors and proper-motion uncertainty also
limit an astrometric solution.

Save the solved image and matched table
----------------------------------------

Choose the combined image you measured, attach the fitted WCS, and keep units,
errors and masks in the FITS interchange::

    >>> image = read_framedata(inputs / 'combined.fits')

    >>> # Attach coordinates to existing pixels; assigning WCS does not resample the image.
    >>> image.wcs = solution
    >>> output.mkdir(exist_ok=True)
    >>> image.write(output / 'astrometric.fits', overwrite=True)
    >>> matched = Table({'id': standards['id'][usable], 'source_row': nearest[usable],
    ...                  'residual_arcsec': radial[usable], 'used_in_fit': fit_rows[usable]})

    >>> # ECSV metadata needs native Python strings rather than NumPy string scalars.
    >>> fit_reference_ids = standards['id'][fit_rows].tolist()
    >>> matched.meta.update({'input_set': inputs.name, 'image': 'combined.fits',
    ...                      'measurements': 'instrumental.ecsv', 'projection': 'TAN',
    ...                      'association_radius_pixels': 8, 'fit_reference_ids': fit_reference_ids})
    >>> matched.write(output / 'astrometric_matches.ecsv', overwrite=True)

    >>> print('Saved: astrometric.fits and astrometric_matches.ecsv')
    Saved: astrometric.fits and astrometric_matches.ecsv

For a field without approximate pointing or identified counterparts, the
Astropop astrometry.net integration requires an installed solver and suitable
local index files; see :doc:`../../modules/astrometry/index`. That external solver is
not executed by this offline example. Continue with :doc:`../photometry/photometric_calibration`.

.. doctest::
    :hide:

    >>> _temporary.cleanup()
