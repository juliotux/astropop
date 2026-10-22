Match positions to a local catalog
==================================

Associate measured sky positions with a small reference catalog without an
online query. Source IDs label the input catalog; coordinates are decimal
degrees and photometric band names are case-sensitive. Here ``g`` is present
but ``G`` is not::

    >>> from astropop.catalogs import SourcesCatalog
    >>> from astropop.math import QFloat

    >>> reference = SourcesCatalog(ids=['A', 'B'], ra=[10., 11.], dec=[20., 21.],
    ...                            unit='deg', mag={'g': QFloat([12., 13.], [0.1, 0.2])})

Find counterparts within a search radius
------------------------------------------

The first two positions below are close to catalog entries in reversed order.
The third is near A but outside the one-arcsecond search radius. Create small
east/north angular offsets on the sphere, rather than adding arcseconds directly
to a right ascension in degrees::

    >>> from astropy import units as u
    >>> from astropy.coordinates import SkyCoord

    >>> origins = SkyCoord([11., 10., 10.], [21., 20., 20.], unit='deg')
    >>> targets = origins.spherical_offsets_by([0.2, -0.4, 2.] * u.arcsec,
    ...                                        [0.1, 0.2, -0.5] * u.arcsec)
    >>> matched = reference.match_objects(targets.ra.deg, targets.dec.deg,
    ...                                   limit_angle='1 arcsec')

    >>> matched.sources_id().tolist()
    ['B', 'A', '']
    >>> print(matched.magnitude('g').nominal.tolist())
    [13.0, 12.0, nan]

An empty ID and a NaN magnitude identify the unmatched position. Keep matched
rows for subsequent photometric work and display each magnitude with its error::

    >>> valid = matched.sources_id() != ''
    >>> selected = matched[valid]
    >>> magnitudes = selected.magnitude('g')

    >>> for source_id, value, error in zip(selected.sources_id(),
    ...                                    magnitudes.nominal, magnitudes.uncertainty):
    ...     print(f'{source_id}: g = {value:.1f} +/- {error:.1f} mag')
    B: g = 13.0 +/- 0.2 mag
    A: g = 12.0 +/- 0.1 mag

The result preserves input order and length. Matches are independent nearest
neighbors, so repeated inputs can select the same catalog source. It does not
solve a one-to-one assignment or validate photometric compatibility. Supply an
observation epoch when proper-motion propagation is needed; the epoch example
in :doc:`../../modules/catalogs/index` explains the required reference metadata.

Inspect angular separations
----------------------------

Measure the nearest-neighbor distance for every input, including the unmatched
one, before deciding whether the search radius is appropriate::

    >>> indices, separation, _ = targets.match_to_catalog_sky(reference.get_coordinates())

    >>> print('Nearest separations (arcsec):', separation.arcsec.round(2).tolist())
    Nearest separations (arcsec): [0.22, 0.45, 2.06]

.. plot::
    :include-source: false
    :alt: Two catalog neighborhoods with one-arcsecond circles; two input positions fall inside and one outside the radius.
    :caption: Local spherical offsets from A and B. Blue circles are accepted inputs, the red cross is unmatched, and dashed circles show the search radius. East increases to the right in this offset diagram.

    import matplotlib.pyplot as plt
    import numpy as np

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from matplotlib.patches import Circle
    from astropop.catalogs import SourcesCatalog

    catalog = SourcesCatalog(ids=['A', 'B'], ra=[10., 11.], dec=[20., 21.], unit='deg')
    origins = SkyCoord([11., 10., 10.], [21., 20., 20.], unit='deg')
    targets = origins.spherical_offsets_by([0.2, -0.4, 2.] * u.arcsec,
                                           [0.1, 0.2, -0.5] * u.arcsec)

    # The result follows input order and retains an unmatched row beyond the radius.
    matched = catalog.match_objects(targets.ra.deg, targets.dec.deg, '1 arcsec')
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), layout='constrained',
                             sharex=True, sharey=True)
    for ax, index, rows in zip(axes, [0, 1], [[1, 2], [0]]):
        center = catalog.get_coordinates()[index]
        east, north = center.spherical_offsets_to(targets[rows])
        ax.add_patch(Circle((0, 0), 1, fill=False, linestyle='--', color='0.4'))
        ax.plot(0, 0, '*', color='black', markersize=12, label='Catalog position')
        for row, dx, dy in zip(rows, east.arcsec, north.arcsec):
            accepted = matched.sources_id()[row] != ''
            ax.plot(dx, dy, 'o' if accepted else 'x', color='C0' if accepted else 'C3',
                     markersize=8)
            ax.annotate(f'Input {row + 1}', (dx, dy), xytext=(5, 7),
                         textcoords='offset points', fontsize='small')
        ax.set(title=f'Neighborhood of {catalog.sources_id()[index]}',
                xlabel='East offset (arcsec)', xlim=(-1.5, 2.7), ylim=(-1.4, 1.4),
                aspect='equal')
        ax.grid(alpha=0.2)

    axes[0].set_ylabel('North offset (arcsec)')
    axes[0].legend(loc='lower right', fontsize='small')

Each panel has its own catalog position at the origin; it is not a full-field
sky chart. A radius accepts a positional association, not proof of source
identity. In real data inspect separation distributions, crowding and coordinate
uncertainties before widening it. A source can have a nearby neighbor and still
remain unmatched under the chosen limit, as input 3 does here.

Save a reusable snapshot
-------------------------

Write a table with explicit units and provenance. Reading a table does not
recreate a SourcesCatalog automatically; reconstruct the required fields.
This minimal snapshot keeps IDs, coordinates and one band, not arbitrary
auxiliary query fields or a complete proper-motion model::

    >>> from astropy.table import Table
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory

    >>> with TemporaryDirectory() as directory:
    ...     filename = Path(directory) / 'reference.ecsv'
    ...     reference.table().write(filename, format='ascii.ecsv')
    ...     saved = Table.read(filename, format='ascii.ecsv')
    ...     restored = SourcesCatalog(ids=saved['id'], ra=saved['ra'], dec=saved['dec'],
    ...                               unit='deg',
    ...                               mag={'g': QFloat(saved['g'], saved['g_error'])})
    ...     print('Restored IDs:', restored.sources_id().tolist())
    ...     print('Right ascensions (deg):', restored.ra().tolist())
    Restored IDs: ['A', 'B']
    Right ascensions (deg): [10.0, 11.0]

For observational reuse also save catalog release, query region, filters,
coordinate frame/epoch, retrieval date and software versions. Do not infer
those from a filename. Continue with :doc:`../../modules/catalogs/selection` for live-service
choices and failure diagnosis.
For a reference with velocities, use the
:ref:`complete coordinate-model snapshot <catalog-motion-snapshot>` example
to retain its epoch and motion metadata.

Match detections measured from an image
---------------------------------------

Apply the same matching interface to measured image positions. This section
uses the :doc:`shared tutorial archive <../index>`: the reference image has a WCS
fitted in :doc:`astrometric_calibration`, and ``standards.ecsv`` supplies local
reference positions and magnitudes. Keep optional arrays enabled before loading
frames as described in :doc:`../index`. Detection and matching remain separate
steps; a WCS is required to convert image centroids into sky coordinates.

Run from the extracted archive root:

.. code-block:: python

    from pathlib import Path

    data = Path('data')

.. doctest::
    :hide:

    >>> from tutorial_data import generate_dataset

    >>> _inputs = TemporaryDirectory()
    >>> data = Path(_inputs.name).resolve() / 'data'
    >>> generate_dataset(data)

Load the image and reference catalog, detect sources, and measure their fluxes
with a local sky annulus::

    >>> from astropop.framedata import read_framedata
    >>> from astropop.photometry import background, daofind, aperture_photometry

    >>> image = read_framedata(data / 'reference' / 'astrometric.fits')
    >>> standards = Table.read(data / 'standards.ecsv')
    >>> field_catalog = SourcesCatalog(ids=standards['id'], ra=standards['ra'],
    ...                                dec=standards['dec'], unit='deg',
    ...                                mag={'R': QFloat(standards['R'], standards['R_error'])})
    >>> sky, noise = background(image.data, box_size=32, mask=image.mask)
    >>> detections = daofind(image.data, threshold=5, background=sky, noise=noise,
    ...                      fwhm=4.24, mask=image.mask)
    >>> measured = aperture_photometry(image.data, detections['x'], detections['y'],
    ...                                r=6, r_ann=(9, 13), bkg_error=noise,
    ...                                mask=image.mask, pixel_flags=image.flags)
    >>> positions = image.wcs.pixel_to_world(measured['x'], measured['y'])
    >>> counterparts = field_catalog.match_objects(positions.ra.deg, positions.dec.deg,
    ...                                            limit_angle='0.5 arcsec')
    >>> usable = (counterparts.sources_id() != '') & (measured['flags'] == 0)

    >>> print(f'Detected: {len(detections)}; matched with zero flags: {usable.sum()}')
    Detected: 9; matched with zero flags: 9

The reference follows the simulation's exact coordinate model. Real reference
errors and imperfect WCS solutions affect these associations; inspect separations
and duplicate identifiers before treating a counterpart as a calibration star.
The measured flux is still instrumental, not a calibrated magnitude.

Export the selected measurements alongside their reference IDs and R magnitudes::

    >>> measurements = measured[usable].copy()
    >>> measurements['catalog_id'] = counterparts.sources_id()[usable]
    >>> magnitudes = counterparts.magnitude('R')
    >>> measurements['reference_R'] = magnitudes.nominal[usable] * u.mag
    >>> measurements['reference_R_error'] = magnitudes.uncertainty[usable] * u.mag
    >>> measurements.meta.update({'catalog': 'bundled synthetic standards',
    ...                           'match_radius_arcsec': 0.5,
    ...                           'image': 'reference/astrometric.fits'})
    >>> output = data.parent / 'reduced'
    >>> output.mkdir(exist_ok=True)
    >>> measurements.write(output / 'matched_measurements.ecsv', overwrite=True)

    >>> print('Saved matched measurements:', len(measurements))
    Saved matched measurements: 9

Continue with :doc:`../photometry/photometric_calibration` to estimate a zero point from
suitable standards. A shared name or band does not alone establish compatible
passbands, epochs or quality.

.. doctest::
    :hide:

    >>> _inputs.cleanup()
