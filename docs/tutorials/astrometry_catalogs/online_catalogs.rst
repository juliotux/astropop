Query a reference catalog and reuse the result
==============================================

Choose a reference survey, query only the required region and bands, inspect
the returned values, then save the response for repeated local matching. Use the
returned sources as candidate references; assess their quality before fitting
a photometric calibration.

Only the packaged-help example below is an offline doctest. Live query recipes
need network access and their rows change with the services; documentation
builds do not execute them. Use :doc:`local_catalog_matching` to practice the
matching and snapshot steps entirely offline.

Inspect the available catalog and bands
---------------------------------------

VizieR configurations are packaged with Astropop. Read their help before
constructing a catalog; construction performs the remote query::

    >>> from astropop.catalogs import vizier

    >>> print(vizier.ucac4.help().splitlines()[0])
    ucac4: UCAC4 Catalogue (Zacharias+, 2012)

``print(vizier.list_vizier_catalogs())`` lists all packaged configurations.
Use the exact band spelling printed by the selected catalog's help. In this
example, UCAC4 offers APASS ``'B'`` and ``'V'`` photometry; survey bands with
similar names are not interchangeable calibration standards.

Query VizieR by coordinates
---------------------------

Using an explicit coordinate avoids an additional object-name lookup. Replace
the center and radius with those of your observation, allowing for the full
footprint rather than only its center pixel:

.. code-block:: python

    from astropy.coordinates import SkyCoord
    from astropop.catalogs import vizier

    center = SkyCoord(ra=2.71675, dec=-54.29066, unit='deg', frame='icrs')
    radius = '10 arcmin'
    catalog = vizier.ucac4(center, radius, band=['B', 'V'])

    print(f'Returned sources: {len(catalog)}')
    print('Requested filters:', catalog.filters)
    print(catalog.table()[:5])

Inspect the actual count and columns before accessing a row. An empty result
differs from a failed request; keep timeout or service errors visible. The
shared ``table()`` contains normalized IDs, coordinates and magnitudes.
``query_table`` retains the service-specific auxiliary columns. Check survey
quality, missing photometry and completeness before choosing standards.

Save the query and its provenance
---------------------------------

Save the normalized table for local work and retain the original response for
quality/provenance information. Reading the normalized table alone does not
restore the catalog's full coordinate epoch or proper-motion model.

.. code-block:: python

    from pathlib import Path
    from datetime import datetime, timezone

    output = Path('catalogs')
    output.mkdir(exist_ok=True)
    snapshot = catalog.table()

    # Record query choices and retrieval time alongside the downloaded values.
    snapshot.meta.update({
        'service': 'VizieR', 'catalog': 'UCAC4',
        'center_ra_deg': float(center.ra.deg),
        'center_dec_deg': float(center.dec.deg),
        'radius': radius, 'bands': list(catalog.filters),
        'retrieved_utc': datetime.now(timezone.utc).isoformat(),
    })
    snapshot.write(output / 'ucac4.ecsv', overwrite=True)
    catalog.query_table.write(output / 'ucac4_query.ecsv', overwrite=True)

    print(f'Saved normalized and auxiliary rows: {len(snapshot)}')

These named outputs are overwritten when rerun. Keep a dated copy when a
reviewed catalog snapshot is part of a reproducible reduction.

Match measured positions locally
--------------------------------

The following recipe assumes a measured-source table containing ``ra`` and
``dec`` in decimal degrees, derived from a validated image WCS. It reloads the
saved coordinates and bands without contacting VizieR:

.. code-block:: python

    import numpy as np
    from astropy.table import Table
    from astropop.catalogs import SourcesCatalog
    from astropop.math import QFloat

    saved = Table.read(output / 'ucac4.ecsv')
    reference = SourcesCatalog(
        ids=saved['id'], ra=saved['ra'], dec=saved['dec'], unit='deg',
        mag={band: QFloat(saved[band], saved[band + '_error'])
             for band in ['B', 'V']})
    measured = Table.read('reduced/sky_positions.ecsv')
    matched = reference.match_objects(measured['ra'], measured['dec'],
                                       limit_angle='1 arcsec')
    found = matched.sources_id() != ''
    photometry = matched.magnitude('V')

    # A positional match can still lack a usable magnitude in the requested band.
    usable = found & np.isfinite(photometry.nominal)

    print(f'Positional matches: {found.sum()} / {len(measured)}')
    print(f'Matches with finite V magnitudes: {usable.sum()}')

Unmatched rows stay in the input order, with empty IDs and NaN photometry.
Matches are independent nearest neighbors; multiple detections can select the
same reference star. This minimal reload uses the saved RA/Dec positions
without reconstructing an epoch or propagating proper motion. Proper-motion
columns alone are insufficient: retain their coordinate frame, reference epoch,
units and any distance/radial-velocity information needed for propagation.
For a specific observation epoch, another option is to call
``catalog.get_coordinates(obstime=...)`` on the original catalog and save those
propagated positions with their target epoch. Do not pass a new epoch to the
minimal reloaded catalog and expect its missing motion model to be restored.
See :doc:`../../modules/catalogs/index` for the coordinate and epoch contracts.
The :ref:`coordinate-model snapshot example <catalog-motion-snapshot>` shows
how to preserve a SkyCoord with its epoch and velocities in ECSV.

Use SIMBAD for identity and literature references
-------------------------------------------------

SIMBAD is an alternative for object identity and compiled measurements. Query
the intended bands explicitly, then inspect the references for each measurement:

.. code-block:: python

    from astropop.catalogs.simbad import SimbadSourcesCatalog, simbad_query_id

    sirius = SkyCoord(ra=101.28715, dec=-16.71612, unit='deg')
    literature = SimbadSourcesCatalog(sirius, '10 arcmin', band=['B', 'V'])

    print(f'SIMBAD sources: {len(literature)}')
    print(literature.table()[:5])
    print('V references:', literature.magnitudes_bibcode('V'))

    name = simbad_query_id(sirius.ra.deg, sirius.dec.deg, '10 arcsec',
                           name_order=['HD', 'HR', 'TYC'])

    print('Preferred identifier:', name)

The preferred identifier can be empty if nothing matches. Identifier lookups
make individual remote queries; query a region once and match locally for a
large source list. A compiled SIMBAD magnitude may come from a different
photometric system or epoch than your image. Read its bibliographic source
before using it as a calibration standard.

For Gaia astrometry use :doc:`../../modules/catalogs/gaia`; for selecting a service
and handling timeouts see :doc:`../../modules/catalogs/selection`. The separate
:doc:`../photometry/photometric_calibration` tutorial explains how reviewed, compatible
standards enter the zero-point calculation.
