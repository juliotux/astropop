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
in :doc:`../tools/catalogs` explains the required reference metadata.

Inspect angular separations
----------------------------

Measure the nearest-neighbor distance for every input, including the unmatched
one, before deciding whether the search radius is appropriate::

    >>> indices, separation, _ = targets.match_to_catalog_sky(reference.get_coordinates())
    >>> print('Nearest separations (arcsec):', separation.arcsec.round(2).tolist())
    Nearest separations (arcsec): [0.22, 0.45, 2.06]

.. plot:: _ext/guide_figures.py plot_catalog_matching
    :include-source: False
    :alt: Two catalog neighborhoods with one-arcsecond circles; two input positions fall inside and one outside the radius.
    :caption: Local spherical offsets from A and B. Blue circles are accepted inputs, the red cross is unmatched, and dashed circles show the search radius. East increases to the right in this offset diagram.

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
those from a filename. Continue with :doc:`../how_to/catalogs` for live-service
choices and failure diagnosis.
