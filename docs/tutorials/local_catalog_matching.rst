Match positions to a local catalog
==================================

This tutorial executes with local arrays under pytest. It performs no online
query. Source IDs label the input catalog; coordinates are decimal degrees and
photometric band names are case-sensitive. Here ``g`` is present but ``G`` is not::

    >>> import numpy as np
    >>> from astropop.catalogs import SourcesCatalog
    >>> from astropop.math import QFloat
    >>> reference = SourcesCatalog(ids=['A', 'B'], ra=[10., 11.], dec=[20., 21.],
    ...                            unit='deg', mag={'g': QFloat([12., 13.], [0.1, 0.2])})
    >>> matched = reference.match_objects([11., 10., 180.], [21., 20., -40.],
    ...                                      limit_angle='1 arcsec')
    >>> assert matched.sources_id().tolist() == ['B', 'A', '']
    >>> valid = matched.sources_id() != ''
    >>> selected = matched[valid]
    >>> np.testing.assert_allclose(selected.magnitude('g').nominal, [13, 12])
    >>> np.testing.assert_allclose(selected.magnitude('g').uncertainty, [0.2, 0.1])
    >>> assert np.isnan(matched.magnitude('g').nominal[2])

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
    ...     assert restored.sources_id().tolist() == ['A', 'B']
    ...     np.testing.assert_allclose(restored.ra(), [10, 11])

For observational reuse also save catalog release, query region, filters,
coordinate frame/epoch, retrieval date and software versions. Do not infer
those from a filename. Continue with :doc:`../how_to/catalogs` for live-service
choices and failure diagnosis.
