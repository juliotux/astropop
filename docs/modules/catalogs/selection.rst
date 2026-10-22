Choose and reuse catalog queries
================================

First decide whether you need names, astrometry/proper motion, or photometric
bands. The shared :class:`~astropop.catalogs.SourcesCatalog` interface contains
IDs, coordinates and optional magnitudes; it is not the full Astroquery API.
Learn its matching behavior with :doc:`../../tutorials/astrometry_catalogs/local_catalog_matching`
before depending on a remote service.

.. list-table:: Query decisions
    :header-rows: 1
    :widths: 25 40 35

    * - Need
      - Starting point
      - Check before use
    * - Object identity and literature photometry
      - :doc:`simbad`
      - Object type, band availability, missing magnitudes, coordinate epoch.
    * - Gaia DR3 astrometry and available bands
      - :doc:`gaia`
      - Proper-motion epoch, row completeness and quality columns.
    * - A configured survey or custom VizieR table
      - :doc:`vizier`
      - Packaged table identifier, exact filter spelling and units.

Print :func:`~astropop.catalogs.vizier.list_vizier_catalogs` to inspect the catalog
names/descriptions shipped with your installed version. This reads packaged
configuration, not a live catalog. Each returned catalog exposes ``filters``;
use those names exactly. Missing errors or photometry do not establish a
high-quality measurement. Inspect finite magnitudes/errors and appropriate
survey quality columns before calibration.

Live queries and failures
--------------------------

Examples in the service guides require network access and are intentionally
static. Record the query region, release, filters, magnitude limits, epoch and
retrieval date. Reuse a reviewed ECSV/FITS snapshot for repeated local matching;
see the local tutorial. Preserve auxiliary columns separately when needed.

A service timeout differs from an empty scientific result. Catch the actual
exception from a failed request and retain its diagnostic details; do not
substitute an empty successful catalog. Empty-result behavior and exception
types can differ between services and dependency versions. Inspect the response
length and the service guide before indexing row zero. Verify table/column names
against the installed packaged configuration and service schema before changing
URLs or retrying indefinitely.

Gaia defaults to ``query_mode='async'``, which submits an asynchronous job and
polls for its completion. Select ``query_mode='sync'`` explicitly to avoid that
polling path:

.. code-block:: python

    from astropop.catalogs.gaia import GaiaDR3SourcesCatalog

    catalog = GaiaDR3SourcesCatalog(
        (101.28715, -16.71612), radius='20 arcsec', query_mode='sync')

Synchronous queries are paginated to retrieve complete results; they still
have network timeouts and service limits. See :doc:`gaia` for
pagination and failure behavior. Restrict the query region to the scientific
need, use bounded retries in the application, and keep query failures visible.
A successful recorded-response test checks our parser, not the remote service's
current health.

For high-proper-motion targets, propagate reference coordinates to the observation
epoch with available motion metadata before interpreting separations. Matching
is independently nearest-neighbor within ``limit_angle``; it preserves unmatched
rows and permits repeated reference sources. Neither astrometric proximity nor
a shared band name alone establishes a valid photometric zero-point reference.

.. plot::
    :include-source: false
    :alt: A simulated moving star travels beyond its original one-arcsecond match circle between 2015 and 2025. Its separation from the observation drops below the match radius when the catalog is propagated to 2025.
    :caption: A synthetic source with proper motion of 1000 mas/year in RA*cos(Dec) and 200 mas/year in Dec. The observation is offset by about 0.18 arcsec from its predicted 2025 position. SourcesCatalog.match_objects rejects it without an observation epoch and accepts it with obstime set to 2025; no online query is made.

    import matplotlib.pyplot as plt
    import numpy as np

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from matplotlib.patches import Circle
    from astropop.catalogs import SourcesCatalog

    reference = SourcesCatalog(ids=['Moving star'], ra=[10.] * u.deg, dec=[20.] * u.deg,
                               pm_ra_cosdec=[1000.] * u.mas / u.yr,
                               pm_dec=[200.] * u.mas / u.yr, distance=[100.] * u.pc,
                               radial_velocity=[0.] * u.km / u.s,
                               obstime=Time(2015., format='jyear'))
    years = np.arange(2015, 2026)
    predicted = reference.get_coordinates(Time(years, format='jyear'))
    # Plot only positions; each propagated coordinate has a different epoch.
    track = SkyCoord(ra=predicted.ra, dec=predicted.dec)
    start = track[0]

    # Offset the final-epoch position slightly to simulate a measured source.
    observed = track[-1].spherical_offsets_by(0.15 * u.arcsec, -0.1 * u.arcsec)
    east, north = start.spherical_offsets_to(track)
    dx, dy = start.spherical_offsets_to(observed)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), layout='constrained')
    axes[0].plot(east.arcsec, north.arcsec, '.-', color='0.6', label='Catalog motion model')
    for index, color, year in [(0, 'C0', 2015), (-1, 'C1', 2025)]:
        axes[0].add_patch(Circle((east.arcsec[index], north.arcsec[index]), 1,
                                 fill=False, color=color, label=f'1 arcsec at {year}'))

    axes[0].plot(dx.arcsec, dy.arcsec, 'k*', markersize=12, label='Observed in 2025')
    axes[0].set(aspect='equal', xlim=(-1.5, 12), ylim=(-2, 5),
                xlabel='East offset (arcsec)', ylabel='North offset (arcsec)',
                title='The same source changes position')
    axes[0].legend(fontsize='small', loc='lower right')
    axes[1].plot(years, observed.separation(track).arcsec, 'ko-')
    axes[1].axhline(1, color='C3', linestyle='--', label='Match radius: 1 arcsec')
    for year in [None, 2025]:
        epoch = None if year is None else Time(year, format='jyear')
        matched = reference.match_objects([observed.ra.deg], [observed.dec.deg],
                                           '1 arcsec', obstime=epoch)
        accepted = matched.sources_id()[0] != ''
        label = f'{"No epoch" if year is None else year}: {"matched" if accepted else "unmatched"}'
        axes[1].plot([], [], ' ', label=label)

    axes[1].set(xlabel='Epoch used for catalog coordinates (year)',
                ylabel='Separation from observation (arcsec)', title='Propagate before matching',
                xticks=[2015, 2020, 2025])
    axes[1].legend(fontsize='small')
    axes[1].grid(alpha=0.2)

Increasing the match radius to span ten years of motion would also accept a
larger neighborhood of unrelated sources. Propagate coordinates using available
motion and epoch metadata before choosing the radius; this noiseless motion
model does not include astrometric uncertainty or prove a unique association.
Input and result diagnostics
----------------------------

.. list-table:: Symptoms, checks and recovery
    :header-rows: 1
    :widths: 24 40 36

    * - Symptom
      - Minimal diagnostic
      - Recovery and contract
    * - Catalog request fails or returns no rows
      - Distinguish service exception from successful zero-length result;
        inspect query region, table/column names and saved response.
      - Use a reviewed local snapshot while diagnosing service health. Do not
        turn a timeout into a successful empty catalog. See :doc:`selection`.
