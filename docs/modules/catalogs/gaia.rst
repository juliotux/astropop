.. include:: ../../references.txt

.. |GaiaDR3SourcesCatalog| replace:: `~astropop.catalogs.gaia.GaiaDR3SourcesCatalog`
.. |gaiadr3| replace:: `~astropop.catalogs.gaia.gaiadr3`

Gaia Catalog
============

Gaia measures stellar positions, motions and brightnesses and provides
additional astrophysical measurements and classifications. This adapter queries
Gaia Data Release 3 (DR3), normalizing a selected set of its columns into a
|SourcesCatalog| while retaining the query table for release-specific access.
It does not expose every product in the Gaia Archive. The release content,
selection and limitations are described by `Gaia Collaboration et al. (2023),
A&A 674, A1 <https://arxiv.org/abs/2208.00211>`_.

|GaiaDR3SourcesCatalog| uses the Astroquery Gaia client to query a field when
constructed. Supply a center, radius and optional bands; the |gaiadr3| callable
is an alias for the same class. Name centers such as Sirius can involve name
resolution as well as the catalog request. These live-query recipes are not
executed by documentation builds; the returned rows depend on the service:

.. code-block:: python

    from astropop.catalogs import gaia

    gaia_sources = gaia.gaiadr3('Sirius', radius='20 arcsec')

    print('Sources:', len(gaia_sources))
    print(gaia_sources.table()[:5])

.. code-block:: python

    from astropop.catalogs.gaia import GaiaDR3SourcesCatalog

    gaia_sources = GaiaDR3SourcesCatalog('Sirius', radius='20 arcsec')

    print('Sources:', len(gaia_sources))
    print(gaia_sources.table()[:5])

Only DR3 is configured by this adapter. All |SourcesCatalog| methods are
available, including local access and positional matching. Preserve the release,
query region and epoch with saved data; see :doc:`index` for motion-aware
snapshots and :doc:`selection` for service selection and reuse.

Select ``'G'``, ``'BP'`` or ``'RP'`` by name/list, use ``band='all'`` (default)
for all three, or ``band=None`` to disable normalized photometry. Available
bands do not guarantee finite measurements in every row. The adapter estimates
standard magnitude errors from each ``phot_*_mean_flux_over_error`` value:


.. math::

    \sigma_{mag} \approx \frac{1.1}{SNR_{flux}}

The factor 1.1 approximates ``2.5/ln(10)`` for first-order flux-to-magnitude
propagation. These errors describe the queried mean-flux precision under that
approximation, not all systematic calibration errors. Inspect missing values,
flux signal-to-noise and appropriate release quality columns before weighting
photometric standards.

Limiting Magnitude and Number of Sources
----------------------------------------

Use ``max_g_mag`` to restrict the query to ``phot_g_mean_mag < max_g_mag``;
``None`` (default) applies no G-magnitude cut. This is a selection by brightness,
not a fixed limit on row count, and the cut is applied regardless of which
photometric bands are requested. For example, keep sources brighter than G=10
in a sixty-arcsecond region around Sirius:

.. code-block:: python

    # Apply the G-band limit in the query to reduce the returned source list.
    gaia_sources = gaia.gaiadr3('Sirius', radius='60 arcsec', max_g_mag=10)

    print('Sources:', len(gaia_sources))
    print(gaia_sources.table()[:5])

Synchronous queries
-------------------

To avoid waiting for asynchronous Gaia jobs, select ``query_mode='sync'``:

.. code-block:: python

    # Use numeric coordinates to avoid a separate object-name lookup.
    gaia_sources = GaiaDR3SourcesCatalog(
        (101.28715, -16.71612), radius='20 arcsec', query_mode='sync')

This uses Astroquery's ``Gaia.launch_job``. Results are fetched in pages of at
most 2000 sources, using source IDs to continue each page, then sorted by angular
distance as usual. Searches larger than one page are not silently truncated.
The magnitude filter applies to every page.

The default remains ``query_mode='async'``. Synchronous mode does not submit
asynchronous jobs or fall back to them. It still depends on the Gaia service:
network errors and slow synchronous requests can time out. All pages share the
existing Gaia query timeout; a failed search raises an exception rather than
returning partial results.

Additional Catalog Properties
-----------------------------

Call ``parallax()`` and ``radial_velocity()`` on |GaiaDR3SourcesCatalog| to
retrieve |QFloat| arrays with uncertainties, in milliarcseconds and km/s,
respectively. The arrays follow source order; unavailable measurements can be
NaN. ``parallax()`` returns the measured parallax, not a distance estimate.

.. automethod:: astropop.catalogs.gaia.GaiaDR3SourcesCatalog.parallax
    :noindex:

.. code-block:: python

    parallax = gaia_sources.parallax()

    print('Parallax:', parallax.nominal, parallax.unit)
    print('Parallax errors:', parallax.uncertainty)

.. automethod:: astropop.catalogs.gaia.GaiaDR3SourcesCatalog.radial_velocity
    :noindex:

.. code-block:: python

    velocity = gaia_sources.radial_velocity()

    print('Radial velocity:', velocity.nominal, velocity.unit)
    print('Velocity errors:', velocity.uncertainty)

The methods below expose release-specific indicators. They are not a uniform
set of independent Boolean quality tests. Interpret them with the original
query fields and the DR3 documentation:

.. automethod:: astropop.catalogs.gaia.GaiaDR3SourcesCatalog.phot_variable_flag
    :noindex:

.. code-block:: python

    print('Variability indicator:', gaia_sources.phot_variable_flag())
    print('Original values:', gaia_sources.query_table['phot_variable_flag'])

.. automethod:: astropop.catalogs.gaia.GaiaDR3SourcesCatalog.non_single_star
    :noindex:

.. code-block:: python

    print('Non-single-star field:', gaia_sources.non_single_star())

.. automethod:: astropop.catalogs.gaia.GaiaDR3SourcesCatalog.in_galaxy_candidates
    :noindex:

.. code-block:: python

    print('Galaxy-candidate field:', gaia_sources.in_galaxy_candidates())

``phot_variable_flag()`` currently returns True whenever the stored string is
not ``'CONSTANT'``. This includes unclassified/unavailable values, so True does
not establish detected variability. ``non_single_star()`` and
``in_galaxy_candidates()`` return their stored fields without a common Boolean
conversion or extra validation. Inspect the original values and release meaning
before using any of these methods to reject or select calibration stars.





Reference/API
-------------

.. automodapi:: astropop.catalogs.gaia
    :skip: gaiadr3
    :no-inheritance-diagram:
    :inherited-members:

.. currentmodule:: None

.. py:data:: astropop.catalogs.gaia.gaiadr3

    Alias of :class:`~astropop.catalogs.gaia.GaiaDR3SourcesCatalog`; accepts the same
    constructor arguments and returns the same catalog type.


.. currentmodule:: None
