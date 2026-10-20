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

The first two positions below coincide with catalog entries in reversed order.
The third lies outside the one-arcsecond search radius::

    >>> matched = reference.match_objects([11., 10., 180.], [21., 20., -40.],
    ...                                      limit_angle='1 arcsec')
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
