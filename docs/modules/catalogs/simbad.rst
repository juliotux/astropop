.. include:: ../../references.txt

.. |simbad| replace:: `~astropop.catalogs.simbad.simbad`
.. |simbad_query_id| replace:: `~astropop.catalogs.simbad.simbad_query_id`
.. |SimbadSourcesCatalog| replace:: `~astropop.catalogs.simbad.SimbadSourcesCatalog`

Simbad Database
===============

SIMBAD is the CDS database of identified astronomical objects. It connects
names from different catalogs with coordinates, photometry and bibliographic
references. Its measurements are compiled from several sources, so a returned
position or magnitude is not automatically on a uniform survey/calibration
system. Check the reference and measurement quality for the intended use.
The database design is described by `Wenger et al. (2000), A&AS 143, 9
<https://arxiv.org/abs/astro-ph/0002110>`_.

The examples below make live queries and are not executed by documentation
builds. Use :doc:`selection` for query reuse and failure handling,
and :doc:`index` for the common local data/accessor contracts.

|SourcesCatalog| from Simbad
-----------------------------

Construct |SimbadSourcesCatalog| with a field ``center``, search ``radius``
and optional photometric ``band`` selection. Construction performs the query
and stores its result; subsequent accessor calls operate on that catalog.
The center can be a SkyCoord, a coordinate pair or a name resolved by the
query utilities. Supply a radius with angular units to make the intended
search region explicit. The ``simbad`` callable is an alias for the same class:

.. code-block:: python

    from astropop.catalogs import simbad

    simbad_sources = simbad.simbad(center='Sirius', radius='1 arcmin',
                                   band=['g'])

    print('Sources:', len(simbad_sources))
    print(simbad_sources.table()[:5])

.. code-block:: python

    simbad_sources = simbad.SimbadSourcesCatalog(center='Sirius',
                                                 radius='1 arcmin',
                                                 band=['B', 'V'])

    print('Sources:', len(simbad_sources))
    print(simbad_sources.table()[:5])


Omitting ``band`` leaves photometry disabled (``band=None``). Supply a band
name/list to request those filters, or ``band='all'`` for all configured filters.
Filter spelling is case sensitive: ``'B'`` and ``'g'`` are different selections.
The instance's ``available_filters`` property lists supported names, while
``filters`` lists photometry included in this result. Available filters do not
imply that every source has a finite measurement in each band:

.. code-block:: python

    print('Available bands:', simbad_sources.available_filters)
    print('Queried bands:', simbad_sources.filters)
    print('V magnitudes:', simbad_sources.magnitude('V').nominal)

Use ``coordinates_bibcode()`` to obtain coordinate-reference bibcodes and
``magnitudes_bibcode(band)`` for the selected band's photometric references.
Both return arrays aligned with the catalog's source order. Read those papers
before treating a literature magnitude as a calibration standard:

.. code-block:: python

    print('Coordinate references:', simbad_sources.coordinates_bibcode())
    print('V references:', simbad_sources.magnitudes_bibcode('V'))

The ``query_table`` property returns the table retained by the adapter.
For TAP responses, Astropop normalizes the column names and photometric rows
before retaining it; this is not the complete original service response.
The normalized table and ``table()`` serve different purposes: ``table()``
contains the common identifiers, coordinates and photometry; ``query_table``
includes the retained reference columns. Neither accessor performs a new query.

Identifying Sources in Simbad
-----------------------------

The module-level |simbad_query_id| function searches near supplied decimal-degree
RA/Dec positions and returns a preferred identifier. It accepts a scalar position
or arrays of positions. This is a remote identifier lookup, rather than a method
that searches an already loaded SimbadSourcesCatalog:

.. code-block:: python

    from astropop.catalogs.simbad import simbad_query_id

    print(simbad_query_id(101.287155, -16.716115833333333, limit_angle='1 arcsec'))

    names = simbad_query_id(ra=[101.287155, 101.28876708333333],
                            dec=[-16.716115833333333, -16.716867777777775],
                            limit_angle='1 arcsec')

    print('Matched identifiers:', names)

Supply ``limit_angle`` as the angular search radius. An empty string is returned
when no acceptable name is found, including when requested identifier prefixes
are unavailable. It does not establish that the sky position contains no source.

The default priority begins with SIMBAD's main identifier. Set ``name_order``
to choose identifier prefixes explicitly: this example tries HD, then HR, then
TYC, returning the first available match in that order:

.. code-block:: python

    # Choose the first available identifier following this catalog-prefix order.
    name = simbad_query_id(101.287155, -16.716115833333333, limit_angle='1 arcsec',
                           name_order=['HD', 'HR', 'TYC'])

    print('Preferred identifier:', name)

.. warning::

    Array input is handled with ``numpy.vectorize``. Each position triggers a
    region query, followed by an identifier query when a source is found; this
    is not a single batched service request. For a large source list, query the
    field once and use local matching instead.




Reference/API
-------------

.. automodapi:: astropop.catalogs.simbad
    :skip: simbad
    :no-inheritance-diagram:
    :inherited-members:

.. currentmodule:: None

.. py:data:: astropop.catalogs.simbad.simbad

    Alias of :class:`~astropop.catalogs.simbad.SimbadSourcesCatalog`; accepts the same
    constructor arguments and returns the same catalog type.


.. currentmodule:: None
